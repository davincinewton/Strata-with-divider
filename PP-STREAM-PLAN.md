# PP-STREAM plan — streaming stages (helpers) in the pp split

Status: implemented with this commit (engine flag --pp-stream N, the two-phase placer, the launcher
wiring); the doctrine-rig measurements listed under Validation remain open.
Companion to DIVIDER-PLAN.md.

Principle: **reuse the logic, not the exact code or settings.** Upstream's auto-split
LOGIC (cache fill vs free VRAM, miss mass x link speed, layers x per-card speed, ties to
the balanced placement) is what places helper and Main; the code adapts to our graph
wherever the old assumptions (every card may stream; all layers are free) were baked in.

## Phase A — dividers (arithmetic unchanged)

The divider capacity arithmetic as it runs today — whole pair sets, honest slack, shave
to the chain's beat — applied to the divider stages only; it fixes the head ranges
[0, K) in stage order.

## Phase B — helper + Main, by Strata's auto-split logic

The enumeration logic runs over the residual [K, n_layers) and the remaining cards:
cache fill against each card's free VRAM, miss mass x that card's link price, layers x
that card's measured speed, ties to the balanced placement.  The helper gets what its
link and VRAM actually support — the split the promise arithmetic cannot express.

## Adapting our graph to the reused logic (consequences, not features)

- Which cards are helpers is already known: "gpu_roles" in the config.  The launcher
  passes that fact to the engine at start (the transport is an argument; the concept and
  the setting already exist).
- The pp verification keeps its promise check for divider stages; a helper stage is
  verified the way Main already is.  (Same rule, correct population.)
- The enumeration carries a start layer and a card subset — the same logic, pointed at
  the residual instead of at everything.
- The pp time line already prices Main's streamed sweep; the same pricing covers any
  streaming stage, so the boot numbers stay true.

Default with no helpers in the config: every current path runs exactly as today.

## Not in this plan

Link auto-detection (card identity is not path identity — the config says what a card
is); app/planner wiring beyond passing the role fact; any default flip; changing what
the enumeration optimizes.

## Validation

Build CUDA; boot the standard fleet with no helper in the config and diff the log
against today (must match); two-card mechanism smoke here (--gpu 5,0 --prefill N, GPU5
as helper): table sane, promise ledger clean, prompts run.  Tuning and the ranking
question belong to the rig that has a helper on a good link.
