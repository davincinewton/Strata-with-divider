# DIVIDER-MIGRATION.md — runbook: re-apply the private divider fork onto a new upstream Strata release

Audience: an agent (or person) doing the migration. It assumes you have shell access to
this machine and this checkout (`/home/bz/Strata`). Read it fully before touching git.

Version 1.0 — written after the v0.2.9 → v0.2.12 migration (2026-11-08), which is the
worked example throughout.

---

## 0. What "the divider" is

Private, never-upstreamed fork work that turns a pile of small GPUs into one inference
fleet: whole-layer expert residency ("dividers"), pp-first prefill, Main streaming.

- Engine knobs: `STRATA_SPLIT_PP`, `STRATA_PROMPT_COST_MAIN`, `STRATA_STAGE_TRIM`,
  `STRATA_SPLIT_RELEASE`, `STRATA_SPLIT_OWN`, `STRATA_SPLIT_RING`, `STRATA_HANDOFF_MODE`,
  `STRATA_HANDOFF_DEPTH`, `--pp-stream N`, `mtp_first = false` (drafter always on Main).
- Terminal side: `strata_control/divider.py`, `tests/terminal/test_divider.py`,
  `tests/data/divider_vectors.json`.
- Docs/design: `DIVIDER-PLAN.md`, `PP-STREAM-PLAN.md`, `CRASH-6DIVIDER-DEQUANT-IMA.md`.
- Boot check: `pp-boot-test.sh` (see §5.3). Fleet config:
  `strata-unsloth-ud-q6_k_xl-divider-4096-3-sq.json` (GPUs `1,2,3,4,5,6,0`, context 262144,
  `--resident-experts --prefill 6144 --layer-split auto`).

The fork's changes are ~29 files, ~+4000 lines. Most live in `src/program/generate.cpp`
(one 22k-line file) — that file is where the pain is, every time.

## 1. Asset inventory (update this table every migration)

| What | Name | Role |
|---|---|---|
| Frozen fork on v0.2.9 | branch `divider-sq`, tag `divider-sq-known-good` | rollback rung, never modify |
| Baseline patch v0.2.9 | `/home/bz/divider-on-v0.2.9.patch` | rebuild-from-nothing insurance |
| Fork on v0.2.12 | branch `divider-on-0.2.12` (37 commits on tag `v0.2.12`), tag `divider-on-0.2.12-known-good` | current work + rollback |
| Patch v0.2.12 | `/home/bz/strata-divider-on-v0.2.12.patch` | audit diff + quick apply |
| Older branches | `divider`, `divider01`, `divider-backup-615d3591` | ancestors of `divider-sq`, historical only |

Rules:

1. A promoted branch is **frozen forever** (it is someone's rollback). New migration =
   new branch from the new tag.
2. Never `git commit --amend` or otherwise rewrite a frozen branch.
3. Keep branches + patch files in sync (re-export the patch after any new commit).

## 2. When to migrate

Every 2–3 upstream releases. Each migration costs ~1–2 h (the v0.2.12 one hit ~12
conflict commits in 36). Waiting much longer turns 15 conflicts into 60.
Do it in one sitting; a half-resolved cherry-pick sequence across days is how mistakes
happen.

## 3. Pre-flight

```bash
cd /home/bz/Strata
git fetch --tags origin
git status --short | grep -v '^??'      # must be empty; stash/commit anything else first
ls /home/bz/*.patch                      # both rollback patches must exist
```

Know these machine facts:

- **A production fleet may be running from this checkout.** The engine binary it runs is
  `engine/strata` — untracked/ignored, so git checkouts never disturb it. Building is
  safe (ninja links `build/strata`, a different inode). The **boot test is not**: it needs
  the fleet's GPUs free, i.e. the server stopped. **Ask the user before stopping it.**
- ninja is not on PATH: use `.venv/bin/ninja`. The build tree is `build/` (CMake+Ninja,
  CUDA archs 75+86). Incremental rebuild of the engine: `.venv/bin/ninja -C build strata`.
- Tests use `.venv/bin/python`.
- git here is 2.34.1: no `git merge-tree --write-tree` (don't plan around it).
- Server restart command (old engine):
  `.venv/bin/python -m serve.server --engine strata --config strata-unsloth-ud-q6_k_xl-divider-4096-3-sq.json --port 8081`

## 4. Migration procedure

### 4.1 Branch + cherry-pick the commit sequence

```bash
git checkout -b divider-on-<NEW> v0.<NEW>
git cherry-pick <OLD_BASE>..<OLD_DIVIDER_BRANCH>     # e.g. v0.2.12..divider-on-0.2.12
```

The range form (old base tag .. old divider branch) is the whole fork, no hand-counting.
Cherry-pick **commits, never the big patch file** — one 328 KB `git apply` is all-or-nothing
and unmaintainable to resolve; 37 small commits localize every conflict to one feature and
keep messages/authorship. The `.patch` file is for auditing (and `git apply --3way` as a
shortcut only when you *expect* zero conflicts).

If you must abort mid-sequence: **`git cherry-pick --abort` resets the branch to the
sequence start and loses all applied commits** (objects survive). Recover with
`git reflog` → `git reset --hard <tip>`. Lesson learned the hard way.

### 4.2 Conflict doctrine

- Independent features on both sides → keep **both** (upstream SPILL WALK / DFlash2 /
  `late_alloc` / `uma_refuse_chunk_if_full` / `arm_prompt_helpers` vs the fork's budget
  prints, partner ctx, `--pp-stream`).
- The fork deliberately replaced upstream behavior → take **fork**
  (`mtp_first = false`, `STRATA_STAGE_TRIM` default-on, `ndense_alloc` → `slot0_nd_full`).
- Upstream wrapped fork-relevant code in a lambda/function (e.g. `arm_prompt_helpers`):
  nest the fork content **inside** the wrapper; check for extra call sites first
  (`grep -n <name>` — re-arm call sites are real features, not duplicates).
- Verify any symbol a resolved hunk uses against the **final** fork state
  (`git show divider-on-<OLD>:<file>`), not the mid-sequence one — later commits delete
  things.
- After editing: `grep -c '^<<<<<<<\|^=======$\|^>>>>>>>' <file>` → must be 0, then
  `GIT_EDITOR=true git cherry-pick --continue` (keep messages verbatim).

### 4.3 The silent mis-merge trap — ALWAYS run this check

Git's 3-way merge can drop or duplicate a closing brace while reporting a clean apply.
It really happened (M5-P3 onto v0.2.12): the merge consumed upstream SPILL-WALK-STEP-4's
`}` as context, `main()` never closed, and no conflict was ever shown. Brace *counts*
cannot catch a swap; only a compiler can. Cheap oracle, ~10 s per file:

```bash
# flags copied from build/build.ninja (DEFINES/FLAGS/INCLUDES of the strata target):
g++ -fsyntax-only -std=c++20 -O0 \
  -DSTRATA_HAVE_DFLASH2=1 -DSTRATA_HAVE_QWEN35=1 -DSTRATA_NATIVE_EXPERTS=1 \
  -DSTRATA_PREFILL_FUSED=1 -DSTRATA_PREFILL_MMQ=1 -DSTRATA_Q6K_EXPERTS=1 \
  -DSTRATA_VERIFY_MAX_T=8 '-DSTRATA_VERSION="0.1.58"' \
  -Iinclude -Ithird_party/llama.cpp/ggml/src/../include \
  -isystem /usr/local/cuda/targets/x86_64-linux/include \
  src/program/generate.cpp
```

(`generate.cpp` has no `<<<>>>`, so plain g++ parses it; run the oracle on every touched
`.cpp` too.) Watch for `expected ‘}’ at end of input`.

If it fires, bisect the cherry-pick sequence — each step is one 10 s parse:

```bash
commits=$(git rev-list --reverse <OLD_BASE>..<branch>); lo=1; hi=$(echo "$commits" | wc -l)
# while lo<hi: git show <commits[mid]>:file > /tmp/b.cpp; g++ oracle; broken → hi=mid else lo=mid+1
```

Then `git show <fork's original commit> -- <file>` and diff it against what landed — the
dropped line is in there. Fix with a normal follow-up commit (message says what the merge
ate); don't rewrite history mid-flight.

Known near-miss class: `const` qualifiers the fork introduced that upstream code later
assigns (e.g. `trim_asked = true` in the early split walk) → make non-const, keep the
fork's default-on logic.

## 5. Verification ladder (in this order)

### 5.1 Tests

```bash
.venv/bin/python -m pytest tests/terminal -q -rf      # ~45 min; background it
```

Compare the FAILED set **against the same run on the old divider branch in a clean
worktree** — never against "zero". Baseline comparison setup:

```bash
git worktree add --detach /tmp/cmp-sq divider-on-<OLD>
git worktree add --detach /tmp/cmp-new divider-on-<NEW>
```

Known environmental failures on this machine (as of 2026-11-08, identical on both
branches — do not chase them):

- Fail only when pytest runs from `/home/bz/Strata` itself (untracked `strata-*.json`
  run-configs make discovery resolve the real checkout): `test_discovery.py::GuiLayout::
  test_list_and_the_wizard_agree_on_the_install`, both `test_shared_models_folder.py`
  `*_agree_on_the_install`, `test_no_acquisition.py::ImportGraphTests` (suite-order
  pollution). They pass in any clean `/tmp` worktree.
- Fail everywhere here (TUI/launcher environment): 3 × `test_tuiqa_monitor.py::LightTheme`,
  3 × `test_tui2_023_old_launcher.py::OldLauncherIsToldTheTruth`,
  2 × `test_launch_tui.py` (`ModelStep` recommendation, `PicturesParallelTuning`).

Pass criterion: the new branch's FAILED set ⊆ known-environmental set.

### 5.2 Engine build

```bash
.venv/bin/ninja -C build strata     # ~10-25 min; links build/strata (safe while server runs)
```

Pass = link succeeds. Do not write `engine/strata` while the server runs (ETXTBSY).

### 5.3 Fleet boot test (needs the fleet's GPUs — stop the server, user-approved)

```bash
kill -TERM <serve.server pid>            # children (engine, vision) follow
bash pp-boot-test.sh base-<OLD> engine/strata        # OLD engine = baseline
bash pp-boot-test.sh new-<NEW> build/strata          # candidate
```

Pass criteria (see §worked example: `/tmp/pptest-base-029.log` vs `...-new-0212.log`):

- Both logs reach `slot 0 verified` (script's marker) — no abort, no
  `THE PROMISE IMPOSSIBLE` / `NOTHING SERVED` (the word `SHORT - promise impossible`
  inside the Main's budget line is normal: Main streams by design).
- Placement lines identical: same `layer split budget, CUDA<n>: layers a-b` per card,
  same `pp capacity-greedy`/`pp-first` verdict, same `K=2,4,6,8,10,12` search lines.
- Numeric-normalized diff shows only numbers + at most cosmetic rewordings:
  ```bash
  diff <(sed -E 's/[0-9]+(\.[0-9]+)?/N/g' logA) <(sed -E 's/[0-9]+(\.[0-9]+)?/N/g' logB)
  ```
- Expect small budget shifts from upstream features (v0.2.12 gave +0.5 GiB expert room
  per divider vs v0.2.9 — upstream trim, a *good* sign, not a regression).

### 5.4 Cutover (only if the user asks for it)

```bash
cp engine/strata engine/strata.bak-$(date +%Y%m%d)
cp build/strata build/libstrata_kq.so engine/   # libstrata_kq.so loads from next to the exe!
```

`engine/strata-vision` is a separate build (`build-vision/`, target `strata-vision`) —
rebuild it too if vision should run the new code. Without cutover, `engine/` still holds
the old engine and the branch is just validated, not deployed.

## 6. Freeze and export

```bash
git tag divider-on-<NEW>-known-good
git diff v0.<NEW>..divider-on-<NEW> > /home/bz/strata-divider-on-v0.<NEW>.patch
# integrity gate — must come out byte-identical:
git worktree add --detach /tmp/verify v0.<NEW>
cd /tmp/verify && git apply /home/bz/strata-divider-on-v0.<NEW>.patch
git add -A && git diff --cached divider-on-<NEW> --stat   # empty = identical
git checkout . # (or just) git worktree remove --force /tmp/verify
```

Then: restore the user's working branch, clean every `/tmp` worktree, restart the fleet
(or leave stopped — whatever the user chose), and report with the pass/fail table.

## 7. Incident ledger (learn from these, don't repeat)

| Incident | Root cause | Rule |
|---|---|---|
| Silent dropped `}` (M5-P3 → v0.2.12) | 3-way merge ate STEP-4's brace as context | always run the §4.3 oracle; bisect; diff vs the original fork commit |
| `trim_asked` read-only | fork made const what upstream later assigns | check for assignments to every `const` you introduce in a merge |
| `cherry-pick --abort` lost 31 applied commits | abort resets to sequence start | recover via `git reflog`; prefer a fix-forward commit over amending mid-pick |
| Baseline patch "missing 8 files" | new files are untracked after `git apply` | always `git add -A` before comparing a patched worktree to a branch |
| Python conflict-surgery off-by-one (several) | wrong marker indices | print the exact region first; write the file before any assert that could fail |
| Boot test fought a live server for VRAM | config's 7 GPUs all in use | ask before stopping the fleet; `pp-boot-test.sh` takes the exe path so old/new compare in place |
