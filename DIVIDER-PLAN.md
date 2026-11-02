# Divider — implementation plan (branch `divider`, v1)

Status: v2 for review. Untracked file; delete freely.

## 0. What changed in v2

- Test target fixed: **Qwen3.8-Flash-Next unsloth UD Q6_K_XL** (~169 GB, 6 shards) on this workstation.
- **Base method changed: from "RAM keeps only truth" to "SSD keeps only truth."** The host RAM arena is no longer the source of truth for experts; the pack on SSD is. This is now the foundation the divider stands on, and it is validated **on this rig before any fleet work**.
- Branch created: `divider` (off `main`).

## 1. Idea and thesis

Add a fourth card role, **Divider**: a split stage (typically a cheap card on a
1x riser) **fully resident for a contiguous range of whole layers** — its dense
share + **all** experts of those layers, pinned, never evicted. Steady-state
traffic on its link drops to per-window hand-offs and prompt-chunk residuals
(KB–MB: fine on x1).

Combined with SSD-truth (Section 3), the fleet economics get dramatically
better: host RAM stops being a capacity axis entirely. A 125 GiB box can serve
a 169 GB (or 3× bigger) model: VRAM fleet holds hot + whole resident layers,
SSD holds the truth, page cache is a free fuzzy cache for the rest.

## 2. The rig and the model (measured 2026-11-02)

- Model: `/home/bz/Strata-data/models/unsloth-UD-Q6_K_XL/…-0000{1..6}-of-00006.gguf`
  = 10.9 MB + 682 MB + 54.4 GB + 49.8 GB + 49.4 GB + 14.8 GB ≈ **169 GB**.
- Native pack: `/home/bz/Strata-data/packs/unsloth-ud-q6_k_xl` (`dense.bin`
  1.4 GB bf16, `native_experts.txt`, tokenizer; **no `experts.bin`** — see open item O1).
- Host: **125 GiB RAM**, NVMe root disk (**only 29 GB free** — see risk R8).
- GPU: not visible in this shell right now (driver down); current config names
  1×Main + 8 slots Off — single card today.
- Baseline from the Q5_K_XL log (same rig, previous day class): decode
  **32–34 tok/s**, decode expert cache hit rate **45–58%**, KV streaming
  **94–95%** VRAM hits, current config uses `--kv-resident 32768`, `--kv int8`,
  131K context, MTP `--spec 4`.
- Derived (estimate — confirm in Phase 0A): experts ≈ 167 GB over ~48 layers →
  **≈ 3.4 GB expert-bytes per layer, ≈ 13.5 MB per expert blob** (12288 pairs).
  Whole-layer residency on this model: **24 GB card ≈ 5–6 layers, 16 GB ≈ 3–4,
  12 GB ≈ 2 (too few)** → fleet cards for *this* model are ≥20 GB (used 3090
  class), not riser-3060 class. The fleet thesis survives; the cheap-card floor
  moves with model size. (For flash-class models the floor was much lower.)

## 3. Base method: SSD keeps only truth

**Today (RAM-truth):** the engine materializes the full expert arena in host
RAM (pinned); GPU caches refill from it; `--resident-experts` shrinks it to the
cache complement; spill to SSD is the fallback when RAM is short (this rig at
169 GB / 125 GiB lives in exactly that squeeze).

**Target (SSD-truth):** no full RAM arena. The pack file is the truth; GPU
cache misses read expert blobs from SSD (mmap → page cache as the fuzzy RAM
tier, or direct reads like the PLE path `--ple-io direct`, POSIX worker pool).
Host RAM then holds: pinned staging rings, activations, session state, page
cache. What already exists in the engine (verify each in 0A):

- `STRATA_ARENA_MMAP=1` — maps a native pack's `experts.bin` read-only instead
  of reading it in (DETAILS.md:251).
- The cache-complement plan's **mmap fallback** ("the rest are read from the
  file when lent"; additional-GPU experts "remain on the mmap fallback").
- `--mmap-experts` — canonical packs only (`experts.bin`); native IQ packs
  "say so at start" (MULTI_GPU.md Limits). This pack has no `experts.bin` → **O1**.
- PLE's proven direct-read worker pool is the design precedent for a
  non-cached bulk-read path; the #1389 warning is the precedent for
  measured miss-stream penalties (18% uncovered = −26% on 50K prompts).

**The one number the base method lives or dies on: the miss stream.**
Blob ≈ 13.5 MB; measured decode hit rate 45–58% today. The engine's
miss/lookup log lines must be converted into **sustained MB/s from the miss
stream at target tok/s** and compared against (a) NVMe random-read ≈ 1–3 GB/s
for MB-scale blobs (plenty for modest streams), and (b) **SSD endurance**:
consumer ≈ 600 TBW ⇒ sustained write-equivalent… reads are cheap for TBW, the
real limit is bandwidth and latency — but page-cache hit rate decides both.
Targets for 0A: decode within ≤10% of the RAM-truth baseline; measure
sustained SSD read MB/s and page-cache hit %; the planner then treats
"SSD read budget" the way it treated RAM before.

**Consequences accepted:** startup no longer pre-reads ~167 GB into RAM (warm
GPU caches still fill from SSD per profile — first minutes are slower; keep
the server warm). WDDM pin limits moot (Linux anyway). `--resident-experts`
complement logic still applies to whatever RAM copy exists (ideally none).

## 4. Ground rules (unchanged; survive upstream churn)

- `divider` branch off `main`; `upstream` remote; weekly rebase. MIT.
- **Engine = fixed contract in v1.** No edits to `CMakeLists.txt`, `setup.py`,
  or auto-placement in `generate.cpp`. Explicit `--layer-split K` bypasses auto.
- Additive + opt-in only; defaults byte-identical.
- v1 = Terminal/run-configs; app UI deferred. Linux only.

## 5. Phase 0A — SSD-truth on THIS rig, zero code (do first)

Same single card, the Q6_K_XL model, existing config minus the RAM assumption:

- Try the existing knobs, one variable at a time, same-day runs:
  1. baseline as-is (RAM/spill behavior — record RSS + tok/s)
  2. `STRATA_ARENA_MMAP=1` (if the native pack path accepts it — O1)
  3. `--resident-experts` ± `--resident-budget-gib` (complement shrink)
  4. combinations; capture `cache complement ready: resident X GiB, pinned Y`
     and the "read from the file when lent" shares.
- Measure: RSS high-water, decode tok/s vs baseline, prefill at 87K prompts
  (the log has ready-made 87–93K prompt cases!), SSD read MB/s (`iostat`),
  page-cache behavior (`free`, drop caches A/B).
- **Exit:** a config where RSS is buffers-only (no full arena), decode ≥ 90% of
  baseline, SSD stream inside budget. This is the base method working — fleet
  Phase 0B assumes it.

## 6. Phase 0B — divider physics on a fleet rig (needs hardware)

One Main (healthy link) + ≥1 divider card. Run config (no code):

```
--layer-split <K: divider owns whole resident layers (≥3; for Q6_K_XL: ≥20 GB card)>
--resident-experts   --kv int8   --max-context <sized>   --kv-resident unset (full KV resident)
env: STRATA_STAGE_TRIM=1        env: STRATA_SPLIT_RING small
--vision cpu                    --pipeline-windows 0
```

Checklist: `caches hold Y of Y`; complement RAM down ≈ divider cache; lend
file-fallback 0; order sane (`gpu_order: "as_given"`, hand-written `gpu` wins);
decode ≈ serial-sum estimate; `--split-device 0` bit-exact A/B; same-day auto
baseline (`docs/BENCHMARK_PROTOCOL.md`).

## 7. Phase 1 — planner (`strata_control/divider.py`, pure, vectors-tested)

Inputs: cards (VRAM, SM×clock, link est.), model geometry (expert bytes/layer,
state-type map GDN/QSA/attention → KV bytes/token, blob size), **SSD read
budget** (from 0A), target context, reserves/buffers (1.5 GB @ 2048 chunk;
`SPLIT_PROMPT_VRAM_GB=12`).

Solve K + order + context + RAM/SSD split under:
1. **Residency fit** per divider: `N×3.4 GB(this model) + dense + KV(range,ctx) + buffers + window + reserve ≤ VRAM`
2. **Stage-time balance**: `layers_i / rate_i` ≈ equal (3090 ≈ 1.8× slower/layer than a 4090-class Main — better than the 3060's 4×)
3. **Context ceiling**: `min over cards of KV room / kv-bytes-per-token(range)`; int8 KV first, move K second
4. **New — miss-stream budget (SSD-truth):** non-resident layers' expected miss MB/s ≤ SSD budget, with the divider's whole layers removed from the stream; raise divider share first when the budget binds. (This is the planner's most interesting lever: dividers don't just add VRAM, they **subtract SSD traffic**.)

Ordering: fastest healthy card → stage 0 (PLE layer 1 + picture reader);
dividers middle (never stage 0 unless forced; then fastest-of-weak,
`--vision cpu`, no pipeline windows); exactly one Main last. Never a divider
last. Fallback ladder: divider → Helper → Off, one line each.
Outputs: run config + human plan text. Opt-in only.

Tests: vectors JSON + pytest (`tests/terminal`), covering: no-fit fallback,
QSA-heavy range, miss-budget tie-breaks, ordering traps.

## 8. Phase 2 — role + CLI plumbing (real diffs, all control-plane)

- `gpu_roles.py`: `ROLES += ("divider",)`, recommendation, divider-aware
  `engine_order` (fixes slowest-first trap), warnings; shared vectors file.
- **`serve/server.py` must accept `"divider"` in `gpu_roles`** — verify
  unknown-role behavior first; minimal additive patch.
- `strata launch`: `--gpu-roles 2=divider,…`, Cards-step key, `--plan` text
  (K rationale + context ceiling + SSD budget), `--status`.
- `gpu_fit.py`: whole-layer gate beside 8 GB / 2 GB-per-card rules.
- Docker `GPU_ROLES` passthrough. App UI deferred; verify app tolerates the key.

## 9. Phase 3 — engine patches only if measurements demand (single-seam, additive)

1. Pinned never-evict cache entries for full-resident stages (if churn observed).
2. Ring shrink when a stage is 100% resident.
3. SSD-truth gaps: if `--mmap-experts`/`STRATA_ARENA_MMAP` do not cover native
   GGUF packs without `experts.bin` (O1) — this is the one engine patch the
   base method may **require**; keep it a load-path flag check at one seam.
4. #448 warning logic for resident stages.

Out forever for v1: per-stage `--kv-resident`; auto-placement divider awareness;
`--kv-grow`.

## 10. Phase 4 — measurement and records

Same-day baselines; passed/failed/not-run kept apart. Matrix: RAM-truth vs
SSD-truth × (0, 1, 2+ dividers) × context (8K/32K/128K) — decode tok/s, prefill
87K prompt time, RSS high-water, SSD MB/s + page-cache hits, startup time,
idle link bandwidth. Name cards/links/OS; list what was NOT tested (Windows,
AMD, Intel, CC<7.5, NVLink rigs).

## 11. Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | Miss stream exceeds SSD bandwidth at low cache-hit rates (45–58% today!) | SSD budget as first-class planner constraint; dividers subtract miss traffic; profile-quality check |
| R2 | Page-cache churn: 169 GB table vs 125 GiB RAM thrash | measure in 0A; direct-read variant; keep hot tail pinned |
| R3 | Native pack has no `experts.bin` → mmap paths may not apply (O1) | 0A discovery; Phase 3 patch #3 if needed |
| R4 | 3.4 GB/layer → this model needs ≥20 GB divider cards; 12 GB risers can't whole-layer it | planner gate + honest Off; flash-class models keep the cheap-riser thesis |
| R5 | Slow-card + slow-link double constraint | stage-time balance in solver; recommendation says Off |
| R6 | QSA-heavy range clamps rig-wide context | state-type-aware K; int8 KV |
| R7 | 8-stage cap (`SplitDrive::kMax`) | design within 7 helpers + Main |
| R8 | **Disk: 29 GB free on /** — model already eats the volume; no room for arena scratch, caches, logs | free space before 0A; any scratch-requiring mode is a non-starter — SSD-truth helps (no arena) |
| R9 | Startup: cache fill + first prompts read GB from SSD over the links | warm server; measure fill time per card/link |
| R10 | Checkpoint/cell identity changes (kv mode, context, kv-resident flips) | re-plan = new sessions; document |
| R11 | Upstream churn (~125 commits/day) | additive-only; zero engine diff unless forced; weekly rebase |

## 12. Milestones

1. **M0A:** SSD-truth config green **on this rig** with Q6_K_XL (≤10% decode loss). Gate.
2. **M0B:** divider physics green on a fleet rig (needs ≥20 GB weak card(s)).
3. **M1:** planner + vectors, usable via hand-written configs.
4. **M2:** `--gpu-roles 2=divider --plan` works end to end (launch, server, status).
5. **M3:** measurement matrix in-branch; go/no-go on Phase 3 patches.
6. **M4 (optional):** app UI, or upstream the pure-policy pieces (fork insurance).

## 13. Open items / questions

- **O1:** does `STRATA_ARENA_MMAP` / `--mmap-experts` apply to this native pack
  without `experts.bin`? (First experiment of 0A; may add the one engine patch.)
- O2: exact GPU(s) on this box + link widths when the driver is back (`nvidia-smi -L`, per-card `lspci`).
- O3: fleet hardware: which cards, how many, risers off one slot? (0B blocker.)
- O4: max-context target for fleet sizing (current rig runs 131072 — keep?).
- O5: naming `divider` vs `resident`/`full` (engine docs already say "resident experts").
- O6: re-generate expert profile for Q6_K_XL? (`data/expert-profile.bin` — which quant was it profiled from? Cache fill quality depends on it.)
