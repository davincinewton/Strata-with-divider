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

`--kv-resident 0` (same 11-layer config): prefill 633.9 (unchanged - the cliff
is expert traffic, not KV), decode 31.8 (-6%): fleet keeps `--kv-resident` at
or above the wanted context; 0 is workable, not free.

"another GPU" audit (resolved, static): the phrase names the OPT-IN peer
expert tier (`src/core/peer_experts.hpp`, `peer_device = -1` default - never
enabled by any config here); no card-to-card expert read happens in these runs.
The PCIe half is real and is the cliff's channel: `cudaHostGetDevicePointer`
device aliases of the pinned arena (expert_source.cpp:6135+, "#588") let a GPU
read host experts by DMA itself - what the divider stages do during prefill.

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
