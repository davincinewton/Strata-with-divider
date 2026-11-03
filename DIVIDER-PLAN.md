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

## 2. The rig and the model (updated 2026-11-02: fleet hardware revealed)

- Model: `/home/bz/Strata-data/models/unsloth-UD-Q6_K_XL/…-0000{1..6}-of-00006.gguf`
  = 10.9 MB + 682 MB + 54.4 GB + 49.8 GB + 49.4 GB + 14.8 GB ≈ **169 GB**.
- Native pack: `/home/bz/Strata-data/packs/unsloth-ud-q6_k_xl` (`dense.bin`
  1.4 GB bf16, `native_experts.txt`, tokenizer; **no `experts.bin`** — see open item O1).
- **Geometry (exact, from `native_experts.txt`):** 48 layers × **512 experts**,
  expert blobs **4,428,800 B**, layer total **2,267,545,600 B = 2.11 GiB/layer**,
  experts total **109.2 GB** — RAM-truth arena FITS the 125 GiB host, so the
  first fleet run stays RAM-truth (0A/0B kept apart). Supersedes the v2 estimate
  (was 3.4 GB/layer, 13.5 MB blobs).
- **GPUs (9; user scan + links):** G0 2080 Ti 22 GB **x16 = Main**, G1 RTX 3090
  24 GB **x1 = divider** (the x1 card IS the experiment), G3 2080 Ti 22 GB
  **link unknown = divider** (confirm at start from the engine's per-stage
  `link … GB/s` lines; rule: the best non-Main link leads stage 0), G2+G4–G8
  RTX 3060 12 GB = idle (3060-on-riser = the future fleet tier; this model's
  whole layer is 2.11 GiB, so 12 GB cards take max ~3 layers with tight margins).
  Both 22 GB "2080 Ti" are modded boards. CC 7.5 (Turing) = supported minimum.
- **First test M0B config (created):**
  `strata-unsloth-ud-q6_k_xl-split-test.json` + `run-unsloth-ud-q6_k_xl-split-test.sh`
  — engine order `gpu: [3, 1, 0]` (`as_given`): **ordering axis is LINK, not
  speed** — x1 cards never lead stage 0 (entry feed + PLE + picture-reader
  duties); the x1 3090 sits MIDDLE where a divider belongs (per-window
  hand-offs only). `layer_split: "5,11"` → stage0 G3 L0–4 = 10.6 GiB resident,
  stage1 G1(x1) L5–10 = 12.7 GiB resident, Main G0 L11–47 (78 GB experts, hot
  cache only). Startup fill for G1: 12.7 GiB over x1 ≈ 10–15 s — one-time, OK.
  (Correction logged: v1 config wrongly led with the x1 card ordering by
  speed; quantified note: forced x1 entry is only ~10 KB/token, still not the
  x1 killer — expert-miss streams are — but the rule stands.)
  `STRATA_STAGE_TRIM=1`; `--kv-resident 32768` kept from the working
  single-card config **as a known deviation** — with G1 confirmed x1, dropping
  to `--kv-resident 0` per plan is next-run tuning once stage 0's link (G3) is
  known. Port 8081 (single-card server keeps 8080), CPU vision kept.
- Host: **125 GiB RAM**, NVMe root disk (**only 29 GB free** — see risk R8).
- Baseline from the Q5_K_XL log (same rig, previous day class): decode
  **32–34 tok/s**, decode expert cache hit rate **45–58%**, KV streaming
  **94–95%** VRAM hits, 131K context, `--kv int8`, MTP `--spec 4`.
- **Blocker (2026-11-02 11:xx):** kernel module 595.91.07 loaded but
  `/dev/nvidia*` nodes missing, `nvidia-modprobe` not installed, `sudo` broken
  in this shell (`/etc/sudo.conf` owner). Root action needed, e.g.
  `apt install --reinstall nvidia-modprobe && nvidia-modprobe -u -m -c 9`, or a
  reboot (udev creates the nodes). Then preflight O2 before launch.

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

## 5. Phase 0A — SSD-truth experiments, RESULT 2026-11-02 (single card, Q6_K_XL)

Same-day runs (driver at /tmp, configs `strata-unsloth-ud-q6_k_xl-m0a-r{1..4}.json`,
port 8081; r4 NOT run — deterministically refused for the same disk reason as r2):

| run | config | decode | host RAM after load | notes |
|---|---|---|---|---|
| r1 | baseline (RAM-truth) | **32.9 tok/s** | RSS 111.1 GB (anon 108 GB), avail 13 G | = the missing **same-day single-card Q6 baseline** |
| r2 | + `STRATA_ARENA_MMAP=1` | 34.3 (noise band) | unchanged | **O1 RESOLVED**: mmap bootstraps — first start WRITES `experts.bin` next to the GGUF shards (needs size+2 GiB free), later starts MAP it (file-backed, page-cache fuzzy tier = the true SSD-truth design). REFUSED here: 28.6 GiB free on the shard drive, needs 103.8 (R8 real) → ran as r1 |
| r3 | + `--resident-experts` | **25.1 tok/s (−24%)** | "used" 6 G + 118 G cache — but complement is 91.9 GiB **page-locked** (mapped pinned); avail only 13→22 G | resident mode is NOT SSD-truth; it trades anon RAM for locked file-mapped RAM and costs a quarter of decode on this rig; 0 file blob reads |

**Conclusions.** (1) Full SSD-truth is disk-gated on this rig: needs a ~104 GiB
free drive co-located with the shards; mechanism itself is confirmed to exist
and self-diagnose (expert_source.cpp:6108-6132). (2) On this machine the base
method stays RAM-truth; planner treats SSD-truth as a conditional it enables
only when `free(next to shard) ≥ experts + blob + 2 GiB`. (3) `--resident-experts`
stays OFF by default on this hardware (−24%); revisit only when a model
exceeds RAM + cache, where it is the alternative to spilling. (4) The
planner's miss-stream budget inputs from here: single-card Q6 hit path 3250
resident experts, adaptive tier active.
M0B's 34.6–35.6 tok/s is now +5.7% vs this same-day single-card baseline.

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

## 12.5 M0B EVIDENCE — first divider runs, 2026-11-02, PASSED

Links verified live: G0 **x16** (engine probe 10.9 GB/s), G1 3090 x1, G3 x1
(all gen1-trained, probe 0.4 GB/s). Engine enumeration: server sets
`CUDA_DEVICE_ORDER=PCI_BUS_ID` + `CUDA_VISIBLE_DEVICES` from config `gpu` —
CUDA numbering follows the config order, NOT nvidia-smi indices by name and
NOT CUDA's default enumeration (which on this board lists the 3090 first;
this caused a device-mapping scare — verify mapping only from the engine's
own probe/CC/free-VRAM lines).

Arrangement A (`gpu [3,1,0]`, `"5,11"`, `--vram-reserve-later-mib 1229`):
- Full residency both dividers: `pre-filled 2560 of 2560` (G3, L0-4) +
  `3072 of its 3072 profiled pairs` (G1 x1 3090, L5-10, 12.67 GiB). x1 fill
  cost 32.8 s/card at 358 MB/s, one-time.
- `token graph hit path: 8987 resident experts` = 2560+3072+3355 exactly;
  `R4 hit path ON` — resident experts compute on-device, zero link traffic.
- Decode **34.6–35.6 tok/s** steady (800 tok, drafts ~76%) vs single-card Q5
  baseline 32–34 → **x1 dividers cost nothing; Q6_K_XL (169 GB) runs at
  single-card-class speed.**
- Decode expert-cache hit **78–79%** (single-card: 45–58%). KV-stream hits
  99.8–99.9%. Main VRAM warning fixed by the reserve (239→767 MiB free;
  uncounted ~557 MiB was HC_Q8 staging + borrowed prompt buffers).

Lead-choice A/B, same day — 2080 Ti lead vs 3090 lead: **TIE**

| | A (G3 lead, 5 lay) | B (3090 lead, 6 lay) |
|---|---|---|
| decode warm 800 tok | 34.5 tok/s | 33.9 / 34.2 tok/s |
| prefill 16,455 tok (2 chunks) | 636.8 tok/s | 637.6 tok/s |
| lead spare VRAM | 3347 MiB | 3064 MiB |

Decode Δ−1% < draft-acceptance noise (66–77%). Prefill equal to 0.1%: x1
lead link carries ~6.5 MB/s of chunk entry at 637 tok/s vs ~400 MB/s probed
→ **lead choice between equal-link dividers is measurement-irrelevant; the
only real asymmetry is VRAM margin (lead carries 5.2 GiB prompt buffers).**
B (`[1,3,0]`, `"6,11"`) is the live config per user choice.
Open next: `--kv-resident 0` pass; same-day single-card Q6 A/B baseline;
"read … or from another GPU" wording audit.

## 12.6 PLANNER-DRIVEN FLEET, 2026-11-02 evening: decode WIN, prefill cliff found and measured

The whole fleet driven by the M2 code (`--gpu-roles 1=divider,3=divider,0=main`
-> planner -> `[3,1,0]` + explicit split + `STRATA_STAGE_TRIM=1`; a config whose
`gpu_roles` carry `"divider"` serves as designed):

| split (dividers first) | decode | prefill 16K | divider residency |
|---|---|---|---|
| 5+6 = 11 (manual-equivalent) | 33.9 | **638.6** | 2560/2560 + 3072/3072 |
| 7+8 = 15 (planner-balanced) | **40.8 / 39.7** | 140.6-152.1 | 3584/3584 + 4096/4096, 11012 resident experts |
| 5+8 = 13 | - | 243.4 | - |

**Decode: the planner's balanced split is +15-18% over the manual one**
(40.8 vs 34.5 tok/s same evening) - whole layers on the x1 cards plus a shorter
Main stage pay exactly as the stage-time rule predicts, links still idle.

**The prefill cliff is real, reproducible and quantified**: above 11 total
whole layers on x1 dividers, long-prompt prefill re-streams about 1 GiB per
divider layer per 8192-chunk over the x1 link although the stage reports 100%
resident (`4096 of its 4096 profiled pairs`, `R4 hit path ON`). `nvidia-smi
dmon` during a 16K prompt: GPU1 rx **404 MB/s sustained** (the x1 link, fully
saturated) while GPU3 idles; the same prompt at 11 divider layers moves ~0.
Byte math matches (+10..25 s per layer past 11 at ~0.4 GB/s). With layer-major
forced on (Main later-reserve 1229->3277 MiB fits the super-chunk): Main's
segment hits 1046 tok/s (64 GB read once per prompt) but the divider stages
STILL stream 761/1137 blobs (39 s on the 3090's x1) - not a chunk-major-only
re-stream. Candidate upstream report: `[Engine] fully resident split stages
re-stream expert blobs during long-prompt prefill` (passed: decode 40.8 with
full residency; failed: prefill 142 at 15 x1 layers; not run: other models,
quants, engines - never tested here).

Planner response (measured rule, not a guess): `X1_RESIDENT_LAYERS_SAFE = 11`
warns above 11 total x1 layers; `max_x1_layers=` caps the balance for
prefill-heavy fleets; decode-first fleets keep the balanced split and the
warning. Reserve tension measured: 3277 MiB Main later-reserve buys
layer-major prefill and costs cache slots (3332->2847) and decode
(40.8->36.8 tok/s).

`--kv-resident 0` (same 11-layer config): prefill 633.9, decode 31.8. CORRECTED
after reading the flag's definition (generate.cpp:605): **0 means "all in
VRAM" - it is the engine default, not "no streaming"**; the run measured the
default against an explicit 32768 and the -6% was noise, not a streaming cost.
Streaming is a positive value under the wanted context. The fleet config's
`--kv-resident 32768` is therefore belt-and-braces for 32K context, removable
without effect.

"another GPU" audit (resolved, static): the phrase names the OPT-IN peer
expert tier (`src/core/peer_experts.hpp`, `peer_device = -1` default - never
enabled by any config here); no card-to-card expert read happens in these runs.
The PCIe half is real and is the cliff's channel: `cudaHostGetDevicePointer`
device aliases of the pinned arena (expert_source.cpp:6135+, "#588") let a GPU
read host experts by DMA itself - what the divider stages do during prefill.

**Status after this evening (M0A-M3 done):** planner (`strata_control/divider.py`),
role in all three copies + vectors, launch wiring (`--gpu-roles 1=divider` ->
explicit split + trim env + saved `divider_plan`), server words, 34 tests green.
Polish left: `--plan` shows the roles but not the planner's split text (it is in
the saved `divider_plan` and the engine log); `max_x1_layers` not yet a launch
flag (planner/agent sets it); M4 (app UI) deferred as agreed.

## 12.7 RAM UNDER THE FLEET, 2026-11-02 night: what actually lowers it, measured

RAM-truth fleet (5,11): engine RSS 114.7 GB, system used 111 G, **avail 9 G**;
decode 33.9, prefill 638.6.

`--resident-experts` on the SAME fleet config (complement 68.42 GiB page-
locked, excluding all 8,862 GPU-resident experts - the divider weights ARE
out of the host copy): system used **10 G**, avail **38-41 G**, anon RSS
4.4 GB - but decode **27.1-27.3 (-19%)** and prefill 565 (-11%). Hit rate was
NOT the cost (80-82%, up); the cost is resident-mode's swap path itself:
`STRATA_ADAPT_OVERLAP_RESIDENT=1` changes nothing because overlap is
`off: not on a layer split` regardless (unmeasured upstream question: why
resident-mode swaps cost ~19% when they replace PCIe reads on the same rig;
single-card measured -24%, consistent).

True SSD-truth (mmap arena, page cache as the fuzzy tier) stays disk-gated:
`experts.bin` needs size+2 GiB (~104) on the pack's drive; `/` has 29 G (only
the 158 G Q6 model is on it), the SATA DATA drive has 58 G free of 119. The
pack is relocatable (`--pack` is a path): freeing ~50 G more on DATA (or a
~$15 SATA drive) enables a first mmap start that writes experts.bin from the
GGUF, and every later start maps it.

## 12.8 ENGINE MOD, 2026-11-02 night: SPLIT RELEASE - the divider weights leave RAM

User's finding (`top` after the resident-mode fleet): `buff/cache 116 GB` -
every byte of the model still sat in host RAM as the mapped GGUF views, even
though the pinned complement already excludes the GPU-resident experts. No
setting fixes that (the resident build reads the bytes it copies THROUGH the
mapping; `STRATA_UNBUFFERED_LOAD=1` measured: unchanged, 37 G->115 G cache as
it copies); the release had to be in the engine.

The mod (this branch, opt-in env `STRATA_SPLIT_RELEASE=1`, default path
byte-identical): `FileExpertSource::release_layer_pages` (madvise MADV_DONTNEED
+ posix_fadvise POSIX_FADV_DONTNEED over the layer's three GGUF views, or the
experts.bin span) + the complement build calls it per layer on the in-place
GGUF arm (experts.bin already did this). Releasing a read-only mapping's pages
is always correct - a later read faults them back from the file; the resident
mode's decode reads none of them (R4 hit path), which is why nothing slowed.

Fleet, same 5,11 config, same evening (2080 Ti x16 main + 3090/2080 Ti x1
dividers, Q6_K_XL):

| | resident mode | + SPLIT RELEASE |
|---|---|---|
| engine VmRSS | 118.0 GB | **83.4 GB** |
| engine mapping RSS (RssFile) | 35.8 GiB | **0.5 GiB** |
| `free` buff/cache | 116 G | **80 G** |
| decode / prefill 16K | 27.2 / 565 | **27.7 / 577** |

Divider bytes in host RAM after it: zero - not in the pinned complement
(excludes all GPU-resident pairs), not in the mapping (0.5 GiB left), not in
the page cache (released as the copy passed). The 80 G cache = the 68.4 GiB
pinned complement (main's non-resident experts, by design) + non-expert pages.
`strata launch` with divider roles now emits `STRATA_SPLIT_RELEASE=1` beside
`STRATA_STAGE_TRIM=1` (inert without the resident mode).

Not done (scope for the next engine step): the arena's RAM-truth mode keeps
its full pinned copy - excluding whole layers THERE needs loader range-skips
+ a file-backed `blob_stable` fallback; and the complement's own -19% decode
against the arena stays an open engine question (12.7). Builds: CUDA (75+86)
passed; HIP/SYCL toolchains not on this rig - not built (the new code is
platform-guarded like its neighbours, unconfirmed there).

## 11.9 M5 — the prefill regression, the funding bug, and the pp-first solver (design)

Measured 2025-11-02 (this rig, 18,632-token prompt, engine `divider` branch):

| config | ring | Main PCIe/prompt | GPU3 (middle x1)/prompt | pp |
|---|---|---|---|---|
| arena 5,11 131 K (roles=helper) | 384 | 204 GB | 0.7 GB (hand-offs only) | **650–661** |
| complement 7,13 262 K, own+noborrow | 96 | 488 GB | 5.3 GB | 300.7 |
| same + `STRATA_SPLIT_RING=384` | 384 | 211 GB | **18.9 GB** | 344 |
| complement 7,13 + LOANS + ring 384 | 384 | 279 GB | 29.0 GB (GPU1 34.9) | 140 |

The law (user's, and it fits all five rows): `pp = 8192 / max(t0, A)` where
`t0` = Main's per-chunk time (compute + its own expert bytes / its link) and
`A` = the interval chunks arrive from the last x1 stage. A clean x1 stage
adds ~3-4 s/chunk (pure compute) - `A` never binds. A shaved x1 stage
re-fetches its missing pairs mid-prompt at ~0.45 GB/s and `A` becomes the
whole story (Y1: 8 GB/chunk ≈ 18 s late; loans: both stages → 58 s).
Main's own clause is the ring: ring 96 re-fetches ≈ 2.4x (215 GB/chunk vs
93 GB at ring 384) - that is the 300-vs-650 pair.

**The funding bug.** In no-loan mode `split_pf_mib` (buffers + a ring booked
as 96 slots, generate.cpp:5232-5240) is charged to EVERY stage's cache
reserve via `stage_room`; at 262 K the KV pool leaves a divider no slack, so
the ring is paid in the divider's EXPERTS - the divider promise (zero dynamic
host traffic) dies exactly when a prompt runs. The engine's auto-split search
(5317+) already computes the right quantities (`stage_left` = free − KV at
`max_context` over the range − dense − reserve; measured links; per-layer ms)
but optimizes DECODE window time and bills `pf` to every stage.

**M5 spec (owner's directive: pp first, decode second; `layer_split` default
`auto` = the solver; a number overrides it, engine still verifies):**

- **P1 (funding):** `STRATA_PROMPT_COST_MAIN=1` → charge prompt buffers +
  ring only to the stage that carries the drafter (Main); dividers pay zero.
  Default off (byte-identical); the control plane injects it with the other
  divider envs (`runner.env_add`).
- **P2 (objective):** inside the auto search, `STRATA_SPLIT_PP=1`: hard
  constraint per divider stage: `stage_left(lb,le) ≥ (le-lb) x 512 x
  cost(layer)` (whole-layer residency at context K); maximize layers off
  Main; guard `max divider stage ms <= 0.8 x Main stage ms` (Main must stay
  fed - clause 1 of the law); the decode window time is PRINTED as an
  advisory, never a veto. Verify mode for explicit splits: if any divider's
  pairs exceed its room, warn loudly ("divider will re-stream over x1: drop
  N layers or context").
- **P3 (timing check):** after P1+P2, `A << t0` holds by construction; keep
  the ring >= streaming working set (384 measured) funded from Main (P1);
  validate GPU0 idle < 5% during a 26 K prompt.

Validation suite: the solver must predict max-8 on a 2080 Ti divider at
262 K, infeasibility of 7,13+ring384 with CUDA1 full residency, and `5,11`
feasible at 131 K; the table above is the ground truth it must reproduce.

**M5-live (measured, this rig, engine `02b0ff6a`+):** the funding fix and the
pp-first solver DELIVERED the promise - the boot now picks `K=7,15`, CUDA1 runs
8 layers with **4096 of 4096** pairs and CUDA0 **3584 of 3584**, and per-prompt
x1 traffic is 0.5 GB (stage0) / 0.9 GB (stage1) per prompt - hand-offs only,
zero expert
bytes (the boot-time verification line is the guard). But pp did not reach
parity at first, and the two remaining levers turned out to be the prefill
GEOMETRY and the streaming PIPELINE, not residency:

- **Layer-major serializes the chain.** Its default (on for CUDA splits) fits
  a super-chunk on every stage here, so chunk c+1 starts on stage0 only after
  c left Main: total = sum of stages (one 18,633-token super-chunk: 88.9 s
  cold, 52.9 s warm = 352). The 650-class runs were fast exactly because Main
  had no VRAM for a super-chunk and fell back to chunk-major, which pipelines
  (8192-token windows overlap stages). The divider fleet must set
  `STRATA_PREFILL_LAYER_MAJOR=0`.
- **The pinned-complement stream is a per-BLOB pipeline.** With full residency
  the complement is redundant insurance, and Main pays ~1 ms per blob: its
  ~55k streamed blobs per chunk make chunk-major + complement the worst
  combination (130 tok/s; ring 1024 made it WORSE, 119 - thrash theory dead;
  the blob count, not the ring, is the cost). The 650-class run streamed the
  same miss set through the arena file-cache path at ~5x lower per-miss cost.
  With the solver guaranteeing divider residency, drop `STRATA_SPLIT_RELEASE`:
  arena + chunk-major is the fleet configuration.
  [SUPERSEDED by 11.10: the "~5x per-miss cost" was a geometry confound - at
  equal chunks the tiers tie - and SPLIT_RELEASE stays, it is what keeps the
  model out of anonymous RAM.]

## 11.10 M5 review (the honest matrix, after the 130 tok/s floor)

One table, all runs on this rig (2080 Ti x16 main = GPU0, 3090 x1, 2080 Ti x1;
Q6_K_XL; 18,632-token prompt). Streaming = where Main's cache misses come
from: ARENA (file page cache) or PINNED (resident-experts blob tier).
Geometry = layer-major (whole prompt per stage, serializes the chain) vs
chunk-major (8192 windows, stages overlap). pp warm, tok/s:

| context | streaming | geometry | ring | split     | pp    | note |
|---------|-----------|----------|------|-----------|-------|------|
| 131 K   | arena     | chunk*   | 384  | 5,11      | 650-661 | *forced: Main had no VRAM for a super-chunk |
| 262 K   | pinned    | lm       | 96   | 7,13      | 300.7 | the yesterday number |
| 262 K   | pinned    | lm       | 384  | 7,13      | 341.6 | Y1 - M5 funding improved it |
| 262 K   | pinned    | lm       | 384  | 7,15 auto | 352.4 | M5c: promise kept (4096/4096, 0 streamed blobs) |
| 262 K   | pinned    | chunk 512(auto) | 384  | 7,15 auto | 130   | M5d - the floor: "auto" sized chunks to 512 here |
| 262 K   | pinned    | chunk    | 1024 | 7,15 auto | 119-122 | ring theory dead: cost is per BLOB fetch, more ring = fewer hits |
| 262 K   | arena     | chunk 512(auto) | 384 | 7,15 auto | 129 | tier tie at identical geometry - the "blob tax" was a confound |
| 262 K   | arena     | chunk 8192 | 384 | auto (5,11) | 662.7-667.2 | geometry fix proven (this boot broke the RAM rule - next row is compliant) |
| 262 K   | pinned    | chunk 8192 | 384 | auto (5,11) | 668.3-674.3 | COMPLIANT WINNER: 72 GiB page-locked complement only, dividers stream 0 |
| 131 K   | single GPU (user's own config, GPU0 x16, arena) | - | - | - | 583-590 | THE parity baseline |

Correction (same night): the "pinned tier ~1 ms/blob" claim above was WRONG -
a confound. At identical geometry the tiers tie (pinned@512 130 vs
arena@512 129); every pinned-vs-arena gap in the table was layer-major vs
chunk size. The rule-relevant difference between the tiers is real but
different: WITHOUT `--resident-experts` the engine's default loads the WHOLE
model into private anonymous RAM (measured: Pss_Anon 104.4 GiB, 116 GB used -
breaking the ssd-keep-only-truth rule); WITH it, only the streamed complement
is page-locked (72 GiB at split 5,11; the GPU-resident divider pairs never
occupy host RAM - the M4 skip) and pp is unchanged (668-674). The compliant
fleet recipe is therefore yesterday's tier plus today's geometry:
`--resident-experts` + SPLIT_RELEASE + PROMPT_COST_MAIN + SPLIT_PP +
SPLIT_OWN + auto split (the 8192 chunk is now the engine's fleet default,
not a config line). (`--mmap-experts` without pinning would
pin nothing, but under RAM pressure its pages evict and prompts stall on the
3.8 MB/s SSD reads - not the default choice.)  The complement size follows
the solver's split: 58.98 GiB at 7,13 (yesterday), 72.07 GiB at the 5,11 the
pp-first solver chose at 262 K - pairs moved to RAM as the split moved layers
off the dividers; RAM evidence for the compliant boot: `free` used 11 G /
avail 35 G, engine Pss_Anon 6.0 GiB (was 104.4 when the rule was broken).

**P2b (same night, measured):** the pp objective now prices the sweep, not
just the compute.  A pp placement is scored by `max over stages of`: a
divider pays its layers' compute only (residency makes its prompt bytes the
hand-off); Main pays its compute PLUS one streamed sweep of its non-held
pairs per chunk at its measured link.  And the fleet's chunk default moved
into the engine: a pp fleet with no user `--prefill` runs 8192-token chunks
(a typed `--prefill N` wins; `--prefill auto` still asks the probe).  The
solver then predicted its own choice before any prompt: K=5,11, Main chunk
7027 ms (70.99 GiB sweep + compute) - the boot's trace measured 7255-7866 ms
per 8192 chunk and pp 666-673 tok/s with `--prefill` removed from the config
entirely.  The prediction matches the reality inside ~10%: the algorithm now
sees what the config used to carry.

Review of the implementation against the plan:

- P1 (prompt path funded Main-only): WORKS, with one honest refinement - a
  divider must still pay its chunk's activation staging (boot failure showed
  a 512-token staging chunk not fitting when charged zero), and the SEARCH
  must charge it too (else the solver over-promises: it gave CUDA1 9 layers,
  serve shaved 529 pairs, the new WARNING line caught it - verify mode works).
- P2 (pp-first auto solver): delivers the promise - K=7,15 chosen, 4096 of
  4096 + 3584 of 3584 pairs resident, x1 per-prompt traffic 0.5/0.9 GB, hand-
  offs only. Its objective (minimize the slowest stage's time) is BLIND to
  geometry and streaming pipeline: it cannot see the two factors that now
  dominate pp. Retarget is on hold until the arena 262 K number exists.
- The law `pp = 8192 / max(t0, A)` holds, but t0 is not just compute + bytes:
  the pinned tier costs ~1 ms per streamed BLOB (GPU0 was 77/80 s busy, only
  ~6 s of it 33-layer compute). Arena chunk-major paid ~5x less per miss
  (static from the 650 run's logs; to be confirmed at 262 K).
- Logging gap FIXED: STRATA_TRACE=1 now prints per chunk per stage: tokens,
  ms, experts streamed, and hands-on vs done (prefill.cpp, the pre-existing
  trace line's gate; default byte-identical).
- runner env_add still writes SPLIT_RELEASE - not reconciled until the arena
  262 K verdict (if arena wins, pinned is divider-insurance only and the
  fleet env drops it; the solver's residency is the promise, not the tier).
- My forced `STRATA_PREFILL_LAYER_MAJOR=0` (M5d/e) was right for ARENA and
  wrong for PINNED: it turned 352 into 130. Which geometry to run is a
  property of the streaming tier - the plan's default must follow the tier,
  not the split. The pinned tier is being dropped from the fleet anyway
  (full residency makes it redundant insurance), pending the arena boot.

## 11.11 M5-P3 (the 7-card fleet): the guard was a red herring - the ledger is the design

The owner added six RTX 3060 12 GB (28 SM, 11.63 GiB, x1: probed 0.4 GB/s) as dividers behind the
2080 Ti x16 Main at 262 K.  The boot's own lines: `pp-first finds no placement`, decode-balance
fallback, every divider 75% short, 60-75 s per 8192 chunk re-streaming ~2.2k pairs over x1.  pp
and decode crashed exactly as the arithmetic says they must: the promise costs 2.11 GiB of experts
+ 0.34 dense (priced 0.18 - the recurring `priced 580 took 1418`) + ~2.4-5.6 prompt staging per
card, on an 11.63 GiB card.  `kMinStageLayers=3` (born in #1616 for the borrow-mode chunk clamp,
576 -> 347 tok/s; #2644 refined the ordering) was NOT the blocker: the base search already runs at
min 1 and the 3-layer retry vetoed nothing in these boots.

The owner's rules, implemented fleet-gated (`SPLIT_PP`, defaults byte-identical):

1. **minimum stage = one whole layer** (the min-3 retry is skipped in fleet mode - its disease,
   the borrow-mode chunk clamp, cannot happen when stages keep their own buffers);
2. **infeasible = ask, never break**: `THE PROMISE IMPOSSIBLE` prints a per-card ledger (what a
   whole layer costs there, how many fit) and option 1 - the largest standard context every card
   keeps the promise at, consent = restart with it (measured: the 262 K@8192 boot printed all seven
   cards and offered 196608); nothing is served;
3. **chunk hint, no action**: option 2 names the largest chunk step that would make the promise fit
   (staging is 160 + chunk x 0.66 MiB per card), the exact config edit, and the measured re-sweep
   cost (130 at 512 vs 668-674 at 8192).

Found by this rig and fixed in the same pass: the pp search **never enumerated 5+ stages**
(C(47,6) ~ 10.7M) - a pp capacity-greedy now places them (max-offload and min-each shapes,
time-shaved, verified by `consider()`; planned K=2,4,6,8,10,12 at 262 K@4096).  The serve-side
verify became a hard abort in pp mode - measured working: a 3-layer plan landing 1496/1536 (170
MiB short) refused to serve with the measured budget lines as evidence; plans now carry 3% + 256
MiB + 0.2 GiB/layer slack over the dense under-pricing.  CUDA1-5 then measured `holds 1024 of
1024` - the promise tier itself is flawless.

Two holes the compliant 195.6-tok/s boot exposed, measured: (a) **CUDA0's own cache was never
verified** - it held 517 of 1024 while the stages' loop checked everyone else (same abort now
added for CUDA0); (b) **Main's chunk stepped down 4096 -> 1536 -> 237** at serve: its cache sizing
ate the staging reserve, every micro-chunk boundary re-swept ~9-16k experts, pp 195 against ~550
planned.  (b) is the P1 follow-through owed: reserve Main's prompt staging for the planned chunk
BEFORE sizing its cache.  Rig conclusion (arithmetic, not rules): the six 3060s cannot be pp-fleet
dividers at high context, and where feasible they starve Main below the single 2080 Ti (583) -
the 3090/2080 Ti divider recipe (668-674) stays the config of this fleet until the 3060s earn a
role at ~65 K or get bigger slots.

## 12. Milestones
1. **M0A:** SSD-truth config green **on this rig** with Q6_K_XL (≤10% decode loss). Gate.
2. **M0B:** divider physics green on a fleet rig (needs ≥20 GB weak card(s)).
3. **M1:** planner + vectors, usable via hand-written configs.
4. **M2:** `--gpu-roles 2=divider --plan` works end to end (launch, server, status).
5. **M3:** measurement matrix in-branch; go/no-go on Phase 3 patches.
6. **M4 (optional):** app UI, or upstream the pure-policy pieces (fork insurance).
7. **M5:** pp-first funding (P1) + auto-split solver (P2) + verify mode;
   pp parity with single-GPU on the 262 K divider fleet. Gate: the 11.9 table.

## 13. Open items / questions

- **O1:** does `STRATA_ARENA_MMAP` / `--mmap-experts` apply to this native pack
  without `experts.bin`? (First experiment of 0A; may add the one engine patch.)
- O2: exact GPU(s) on this box + link widths when the driver is back (`nvidia-smi -L`, per-card `lspci`).
- O3: fleet hardware: which cards, how many, risers off one slot? (0B blocker.)
- O4: max-context target for fleet sizing (current rig runs 131072 — keep?).
- O5: naming `divider` vs `resident`/`full` (engine docs already say "resident experts").
- O6: re-generate expert profile for Q6_K_XL? (`data/expert-profile.bin` — which quant was it profiled from? Cache fill quality depends on it.)
