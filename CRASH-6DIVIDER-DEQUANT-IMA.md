# Crash report — 6-divider fleet (7 cards), `dequant launch: an illegal memory access` on big prompts

Tree: `divider` @ `20409b03` (M5-P2b, pre-7-card-fleet). Captured 2026-11-04 from the live fleet.
The running engine binary (`engine/strata`, built 2026-11-04 20:20, BUILD.json src `a22f54c6e3e07cd9`)
carries extra instrumentation not in this tree (`STRATA_RING_TRACE`, `STRATA_HANDOFF_DEPTH`, ring-trace
log lines) — it was built from a different checkout. All code-path claims below are unconfirmed static
analysis against the 20409b03 tree unless marked otherwise.

## Setup that crashes

- Config `strata-unsloth-ud-q6_k_xl-divider.json`, port 8081, Qwen3.8-Flash-Next UD-Q6_K_XL (48 layers, 169 GB pack).
- 7 cards: `gpu [1,2,3,4,5,6,0]` → CUDA0-5 = six 3060 12 GB x1 dividers (2 whole layers each,
  layers 0-2 / 3-5 / 6-8 / 9-11 / 12-14 / 15-17), CUDA6 = 2080 Ti 22.5 GiB x16 Main (layers 18-47, 30 layers).
- `--max-context 262144 --kv int8 --resident-experts --prefill 1024 --layer-split auto`,
  envs: `STRATA_SPLIT_PP=1 STRATA_SPLIT_RING=384 STRATA_SPLIT_OWN=1 STRATA_SPLIT_RELEASE=1
  STRATA_PROMPT_COST_MAIN=1 STRATA_HANDOFF_DEPTH=4 STRATA_TRACE=1 STRATA_RING_TRACE=1`.
- Boot log: Main's 1024-token prompt buffers (2499 MiB) did not fit (1331 MiB free) → **512-token chunks**;
  CUDA6 left with **339 MiB free — "LOW: its window graphs are captured on first use and need room too"**.
- Decode: ~24-27 tok/s, fine. Crash only when a big prompt is fed.

## Reproduction

Two crashes, identical signature:

1. 25,502-token prompt (conversation continuation, pos0=539): died at the 6th 512-chunk round
   (crash at log line 34910, boot #1 of the 7-card split).
2. 14,859-token fresh prompt (this session's test): chunks 0-10 (positions 0-5632) ran clean at
   ~580 ms/chunk across all 7 stages; died the instant the next round of chunks was dispatched to all
   stages (log line 35719).

Engine tail at crash (both boots):

```
strata ring-trace: run_impl entered stage [18,48) c0=0 n=512
dequant launch: an illegal memory access was encountered     (x2)
dequant launch: driver shutting down                         (x5)
```

`dequant launch:` is the `cudaGetLastError()` check after the dequant kernel launch in
`src/kernels/cuda/dequant_bf16.cu` (`launch<T>`, lines 365/381/412), reached from the prompt's dense
projections `Gemm::native` → `dequant_f16` (`src/prefill/gemm.cu:2698` `gm.native(X, w->native_type,
w->native_data, ...)`) → `dequant_bf16.cu:441`.

### What this error means (CUDA semantics)

After any kernel faults with an illegal memory access, the context is **sticky**: the next
`cudaGetLastError()`/API call on that device reports the IMA, and subsequent calls report
`CUDA_ERROR_DRIVER_SHUTTING_DOWN` (exactly what the 5 other stage threads printed). So the dequant
launch is very likely the **reporter**, not the culprit: the faulting kernel is the one that ran
asynchronously just before, on the same device — a cuBLAS GEMM, the QSA prompt attention, a PLE
gather/upload, a handoff H2D/D2H memcpy, or the MTP prefill graph, all launched on the stage stream
around the same moment.

## Ranked hypotheses (unconfirmed static analysis)

1. **Sticky IMA from a sibling kernel on a tight-VRAM Main (CUDA6).** Main holds 30 layers + the MTP
   head (105 MiB) + the 512-token prompt buffers, with 339 MiB left at boot. Each per-chunk
   `on_chunk` callback runs `mtp.prefill` on Main (draft-head graphs, device-to-device copies into
   `pf_dev_`), and every stage re-enters `run_impl` in the same ~20 ms window, so 7 devices launch
   kernels at once. The crash always lands exactly on that all-stages-dispatch moment — consistent
   with one device faulting in a kernel queued during that window and the error being caught at the
   next `cudaGetLastError()` (the dequant launch).
2. **MTP drafter KV ring wrap.** `MtpDrafter::prefill` writes drafter K/V per prompt cell
   (`cell0 + i` positions, `src/core/mtp.cpp:1466+`); the default drafter window is 32768
   (`include/strata/core/mtp.hpp:49`), larger than both crash prompts (25.5k / 14.8k), so a wrap
   needs a smaller ring than the window — plausible only if the ring was sized down for the 3060s'
   VRAM. Check the drafter ring size at boot for this fleet.
3. **Handoff ping-pong race.** The stage hand-off is a two-buffer host ping-pong
   (`m.hand[2]`, `cudaHostAlloc`, `prefill.cpp:1582`), reused after two chunks; the next stage's
   H2D copy is async. With the custom binary's deeper pipeline (`STRATA_HANDOFF_DEPTH=4`, not in the
   20409b03 tree), a later stage could still be H2D-reading a buffer the producer re-writes. An
   async H2D IMA on the reader's device would surface exactly as observed. Not expressible in the
   vanilla tree's depth — needs the custom-build source to confirm.
4. **Streamed-expert ring (384 slots) wrap.** Main streamed ~254 experts per 512-token chunk
   ("254 experts streamed"); 384 slots cover ~1.5 chunks. If the producer can run >1.5 chunks ahead
   of Main, the ring reuses slots Main is still reading. Usually this corrupts values; an IMA needs
   the DMA to target a freed/unmapped range.

## UPDATE (2048-chunk experiment): the stall watchdog dump changes the picture

A re-run with `--prefill 2048` (buffers still auto-stepped to 512 on this rig) did not IMA — it **hung**:
progress stalled at token 3590 for ~100 s, then the server saw the engine exit. The custom build's
stall watchdog fired first and dumped all 285 thread backtraces (log line 37143). Symbolized with
`addr2line -e engine/strata`:

```
stall report: stage "reading the prompt (batched), done up to token 3590" for 60 s; 0 layers served
  main thread (tid 6924):  main -> Prefill::run -> Prefill::run_impl -> cudaStreamSynchronize  (never returns)
  16/16 expert-pool workers parked, 32 DirectFile readers idle on their condvar,
  Stager threads parked on the generation cv, "host idle for 135518 ms"
```

So the wedge is: **the first stage's ([0,3), CUDA0) compute stream never drains while every host thread
is idle** — the GPU stream is waiting on something no host thread feeds. The last stage [18,48) (Main,
own thread) was still completing chunks (3078, 541 ms, 240 experts) right up to the stall line, so the
pipeline is jammed at the HEAD. The engine's own code comments name this exact failure: the
`Stager::start()` comment (src/prefill/prefill.cpp:722) says the ready[]-array-freeing defect under
the old generation was "**the fleet's prompt-pipeline hang and the copy_i32 illegal access, one
defect**". The binary was verified to contain the drain fix (a941ce7f, "draining old generation"
string present), so the hang survives that fix — the remaining gap is one of:

- the `active` counter's check-then-act window in `Stager::work()` (a thread parked on the cv can be
  woken by a new generation's `cv.notify_all()` and re-enter `work()` after `start()`'s drain check
  sees `active == 0`, then read/claim against the grown arrays), or
- the same lifecycle race on the non-growing path, or
- a second defect in the streamed-walk ring itself (issuer/copy-stream bookkeeping; no "issuer stalled
  at entry" trace line fired, so a threaded issuer was not pacing-blocked in this boot).

Also chronic: 11 stall reports across the whole log (tokens 1536 / 3072 / 4608 / 7680 / 12288 /
24583 ...), including one decode-side "verify window: waiting for the expert copies". The IMA crashes
(`dequant launch`, `prefill copy_i32`) are the sticky-error faces of the same defect family: an IMA on
a freed/overwritten buffer (the classic symptom named in the comment: "unwritten pinned buffers read
as ready").

Diagnostic switches now known to exist in this binary (fingerprinted via `strings`):
`STRATA_PF_STEP_SYNC=1` (per-step stream syncs; a stall's report names the exact step/layer),
`STRATA_PREFILL_COPY_EVENT_EVERY`, `STRATA_PREFILL_COPY_WAIT_DEDUPE`, `STRATA_RING_TRACE`.

## What is NOT the cause (checked)

- Chunk-size buffer overflow: `m.T` caps C at the init chunk (512); the last partial chunk is smaller.
- Weights-out-of-range in the dequant kernel itself: it worked for 10-15 identical chunks before.
- MTP window skip logic: window 32768 > both prompts, so no skip at crash time.

## Next steps to pin it (cheapest first)

0. **Re-run the 26k-token prompt with `STRATA_PF_STEP_SYNC=1`** (switch confirmed present in the
   running binary): every step syncs both streams and logs slow steps, and the stall report then names
   the exact step/layer where the compute stream stops draining. One hang gives the answer.
1. Rebuild from this tree (or add to the custom one): run once with `CUDA_LAUNCH_BLOCKING=1` — every
   launch then synchronizes, so the IMA is reported at the **actually faulting** call, not the next
   `cudaGetLastError()`. Expect a few-fold slower prefill; a 6k-token prompt suffices to reach the
   crash window.
2. `compute-sanitizer --tool memcheck ./engine/strata ...` on the same prompt (slow, but names the
   kernel and address).
3. `STRATA_CRASH_BT=1` (generate.cpp:251) — prints a call stack on fatal signals; note the engine
   exits via `std::exit(1)` after the sticky error, so this helps only if the IMA becomes a real
   SIGSEGV (it may, under `CUDA_LAUNCH_BLOCKING`).
4. Bisection probes: (a) 3090-style 2-divider fleet with the same prompt (does it crash?),
   (b) 7 cards with `STRATA_MTP=0`/no `--spec` (kills hypothesis 1's MTP arm),
   (c) 7 cards with `STRATA_SPLIT_RING=1024` (kills hypothesis 4),
   (d) `STRATA_HANDOFF_DEPTH=1` on the custom build (kills hypothesis 3).
5. At crash time, `nvidia-smi` capture: which GPUs show Xid errors in `dmesg` (Xid 79 = uncorrectable
   ECC, Xid 31 = IMA) — that names the faulting card immediately.

## Artifacts

- Log: `strata-unsloth-ud-q6_k_xl-divider.log` (crash #2 at line 35719, crash #1 at line 34910).
- Config: `strata-unsloth-ud-q6_k_xl-divider.json`.
- Rollback state: branch `divider` @ `20409b03`; pre-rollback state kept in `divider-backup-615d3591`
  (tip `615d3591` + WIP commit `509fe83e`).
