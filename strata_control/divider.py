"""The divider planner: whole layers on a slow-link card, ordered by link, sized by three budgets.

A divider is a split stage that holds COMPLETE, PINNED EXPERTS for a contiguous range of whole layers, so its PCIe
link carries no expert misses at all - which is what makes a card on a 1x riser worth adding. The engine has no role
concept (it takes an ordered device list, an explicit `--layer-split K` and an env), so everything here compiles down
to a run config; v1 needs no engine change. Roles are the person's words; the config is the engine's.

The rules (measured: DIVIDER-PLAN.md sections 2 and 12.5):
1. MAIN is the LAST stage by architecture: the output head, the draft layer and the verify of guesses run there.
   Exactly one Main; it takes the healthy link (and the fastest card when more than one is healthy).
2. A card whose measured link is slow (<= SLOW_LINK_GBPS) is a DIVIDER when it can hold whole layers
   (>= min_divider_layers of them after its fixed costs); otherwise it is Helper when its link is tolerable, else Off.
3. Layer counts follow three budgets at once: RESIDENCY (a divider owns only layers it can hold completely),
   STAGE TIME (a stage's layers / its speed ~ equal across stages; the serial pipeline is only as fast as its
   slowest stage), and CONTEXT (each stage holds the whole context's KV for its OWN layers only - when the caller
   gives a per-layer KV cost, the suggested context is the tightest card's). Main takes the remainder.
4. ORDER: stage 0 carries the entry feed, the PLE injection (layer 1) and, by default, the picture reader, so a
   healthy-link card leads whenever one exists; when every helper is slow, the lead is the slow card with the most
   spare VRAM after its resident layers (the lead also pays for its own prompt buffers). Never Main first, never a
   divider last.

Pure functions, no I/O: `strata launch`, tools/strata_mcp.py and the planner CLI import it. The card dicts are
strata_control.gpu_roles' cards plus one optional field: "link_gbps" (a measured host->device rate; absent or None
means "healthy or unknown", which plans as healthy). Tests: tests/terminal/test_divider.py and
tests/data/divider_vectors.json.
"""

from __future__ import annotations

from typing import Optional

GB = 1 << 30
MB = 1 << 20

SLOW_LINK_GBPS = 3.0            # a measured link at or below this is a divider candidate: misses cost more there than the card earns
MIN_DIVIDER_LAYERS = 3          # fewer whole layers and the card is not worth its own buffers (the engine's auto search asks for 3 too)
MIN_STAGE_LAYERS = 2            # the engine's --layer-split needs K rising from 2; a stage never goes below this
X1_RESIDENT_LAYERS_SAFE = 11    # measured (DIVIDER-PLAN.md 12.6; Q6_K_XL 48x2.11 GiB, 16K prompt, chunk 8192): with more than 11 whole
                                # layers on ALL x1 dividers together, long-prompt prefill re-streams about 1 GiB per layer per chunk over
                                # the x1 link although the experts are 100% resident (dmon: rx 404 MB/s, the link is the stage). Decode
                                # keeps improving with more layers (34.5 -> 40.8 tok/s); prefill collapses (637 -> 243 -> 142 tok/s at
                                # 11/13/15). Decode-first fleets keep the balanced split and the warning; prefill-heavy fleets cap it.
MAX_STAGES = 8                  # SplitDrive::kMax in the engine: at most 7 helpers + the Main card
HEAD_DRAFT_BYTES = 840 * MB     # the output head and the draft layer ride on the last card (~0.8 GB, docs/MULTI_GPU.md)
PROMPT_BUFFER_BYTES = int(1.5 * GB)   # a stage's own prompt buffers at the default chunk (docs: "1.5 GB each at the default 2048-token chunk")
WINDOW_BYTES = 360 * MB         # a verify window, the hand-off room and the capture room of the graphs
RESERVE_BYTES = 700 * MB        # the per-card engine reserve at its defaults (measured fleet run: cache auto reserved 700 MiB)
LEAD_EXTRA_BYTES = 256 * MB     # stage 0: the PLE injection rows and the entry feed room (the picture reader on CPU is free)
SAME_SPEED = 1.05               # cards within 5% of each other count as the same card (gpu_roles)


def _vram_bytes(c: dict) -> int:
    return int(float(c.get("vram_gb") or 0) * GB)


def _speed(c: dict) -> float:
    v = c.get("speed")
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 else 0.0


def _link(c: dict) -> Optional[float]:
    v = c.get("link_gbps")
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 else None


def _runnable(c: dict) -> bool:
    return c.get("usable") is not False and _vram_bytes(c) >= int(4 * GB)


def _per_layer(model: dict) -> int:
    e = model["expert_bytes_per_layer"]
    e = max(e) if isinstance(e, (list, tuple)) else int(e)   # the biggest whole layer: residency promises every layer fits
    return int(e) + int(model.get("dense_bytes_per_layer", 0))


def _fixed(role: str, lead: bool) -> int:
    f = RESERVE_BYTES + PROMPT_BUFFER_BYTES + WINDOW_BYTES
    if role == "main":
        f += HEAD_DRAFT_BYTES
    if lead:
        f += LEAD_EXTRA_BYTES
    return f


def _pick_main(cards: list[dict], explicit: Optional[int]) -> dict:
    """The Main card: the named one, else a healthy link first (Main carries every hand-off verdict and the verify
    that prompt chunks wait on), then the fastest, bigger memory and higher number as further tiebreakers."""
    if explicit is not None:
        for c in cards:
            if c.get("index") == explicit:
                return c

    def key(c: dict):
        return (_link(c) is None or _link(c) > SLOW_LINK_GBPS, _speed(c), _vram_bytes(c), c.get("index", 0))
    return max(cards, key=key)


def plan(cards: list[dict], model: dict, roles: Optional[dict] = None, *,
         slow_link_gbps: float = SLOW_LINK_GBPS, min_divider_layers: int = MIN_DIVIDER_LAYERS,
         max_stages: int = MAX_STAGES, max_x1_layers: Optional[int] = None,
         safe_x1_layers: int = X1_RESIDENT_LAYERS_SAFE) -> Optional[dict]:
    """Work out the whole fleet: roles, the engine order, the explicit --layer-split values, the per-card budgets.

    `cards`  strata_control.gpu_roles cards + optional "link_gbps" (measured host->device; None = healthy/unknown)
    `model`  {"n_layers", "expert_bytes_per_layer", optional "dense_bytes_per_layer",
              "kv_bytes_per_token" (per layer, per context token), "context_tokens" (the plan is checked against it)}
    `roles`  optional explicit {"<card index>": "main"|"divider"|"helper"|"off"}; cards left out are auto.
    `max_x1_layers`  whole layers the x1 dividers may hold in TOTAL - cap them for prefill-heavy fleets (X1_RESIDENT_LAYERS_SAFE).

    None when fewer than two usable cards (a lone card is not a split).
    """
    cards = [c for c in cards if not c.get("integrated")]   # an iGPU shares the system RAM and never becomes a stage (gpu_fit)
    if len(cards) < 2:
        return None                                          # nothing to plan a fleet from
    runnable = [c for c in cards if _runnable(c)]
    if not runnable:
        return None
    n_layers = int(model["n_layers"])
    ctx = int(model.get("context_tokens") or 0)
    bpt = float(model.get("kv_bytes_per_token") or 0.0)
    roles = {int(k): v for k, v in (roles or {}).items()}
    warnings: list[str] = []

    mains = sorted(k for k, v in roles.items() if v == "main")
    main = _pick_main(runnable, mains[0] if mains else None)
    if len(runnable) < 2:                                     # one runnable card: still answer, with every other card off
        kind = {c["index"]: "off" for c in cards}
        kind[main["index"]] = "main"
        warnings.append(f"no second card earns a place: run card {main['index']} alone")
        return {"roles": {str(i): kind[i] for i in kind}, "order": [main["index"]], "layer_split": "",
                "stages": [], "context": None, "warnings": warnings,
                "plan_text": "\n".join([f"divider plan: no split - card {main['index']} alone"]
                                       + [f"  note: {w}" for w in warnings])}
    if len(mains) > 1:
        warnings.append(f"more than one card asked for Main: {mains} - {mains[0]} is Main, the rest are Helpers")
    kind: dict[int, str] = {c["index"]: "off" for c in cards}
    kind[main["index"]] = "main"
    for i in mains[1:]:
        if i in kind and kind[i] != "main" and roles.get(i) != "off":
            kind[i] = "helper"

    # ---- 1. classify; fit dividers by residency (whole layers only) ----
    caps: dict[int, int] = {}
    step = _per_layer(model) + (bpt * ctx if bpt and ctx else 0)
    for c in runnable:
        i = c["index"]
        if i == main["index"]:
            continue
        asked = roles.get(i)
        if kind[i] == "helper" and asked not in ("divider", "helper"):   # a demoted second Main stays a helper
            continue
        if asked == "off":
            kind[i] = "off"
            continue
        if asked == "helper":
            kind[i] = "helper"
            continue
        slow = (_link(c) is not None and _link(c) <= slow_link_gbps)
        cap = (_vram_bytes(c) - _fixed("divider", False)) // step if step > 0 else 0
        if asked == "divider" or slow:
            if cap >= min_divider_layers:
                kind[i] = "divider"
                caps[i] = int(cap)
            else:
                kind[i] = "helper" if not slow else "off"
                warnings.append(f"card {i} cannot hold {min_divider_layers} whole layers (room for {cap}): "
                                + ("it is a Helper" if kind[i] == "helper" else "Off - a slow link without whole layers pays for nothing"))
        else:
            kind[i] = "helper"

    stages = [i for i, k in kind.items() if k not in ("off", "main")]
    if not stages:
        return {"roles": {str(i): kind[i] for i in kind}, "order": [main["index"]], "layer_split": "",
                "stages": [], "context": None,
                "warnings": warnings + [f"no second card earns a place: run card {main['index']} alone"],
                "plan_text": "\n".join([f"divider plan: no split - card {main['index']} alone"] + [f"  note: {w}" for w in warnings])}
    if max_stages > 0 and len(stages) + 1 > max_stages:
        demote = sorted(stages, key=lambda i: (_link(next(c for c in cards if c["index"] == i)) or 0, _speed(next(c for c in cards if c["index"] == i))))
        for i in demote[:len(stages) + 1 - max_stages]:
            kind[i] = "off"
            warnings.append(f"card {i} is Off: the engine runs at most {max_stages} stages")
        stages = [i for i, k in kind.items() if k not in ("off", "main")]
    if not stages:
        return None

    # ---- 2. balance: layers proportional to speed, dividers capped at residency, Main keeps the remainder ----
    by_idx = {c["index"]: c for c in cards}

    def rate(i: int) -> float:
        return _speed(by_idx[i]) or max(1.0, _vram_bytes(by_idx[i]) / (8 * GB))   # no speed reading: memory stands in

    if any(_speed(c) == 0 for c in runnable):
        warnings.append("a card has no speed reading: the balance is sized by memory as a stand-in")
    total_rate = sum(rate(i) for i in stages + [main["index"]])
    t = n_layers / total_rate
    n: dict[int, int] = {}
    for i in stages:
        want = max(MIN_STAGE_LAYERS, int(rate(i) * t))
        n[i] = min(want, caps.get(i, n_layers - MIN_STAGE_LAYERS * len(stages)))
    n[main["index"]] = n_layers - sum(n.values())
    if n[main["index"]] < MIN_STAGE_LAYERS:                     # everyone cannot shrink below MIN at once: dividers give back first
        for i in sorted(stages, key=lambda i: -(n[i] / rate(i))):
            if n[main["index"]] >= MIN_STAGE_LAYERS:
                break
            d = min(n[i] - MIN_STAGE_LAYERS, MIN_STAGE_LAYERS - n[main["index"]])
            if d > 0:
                n[i] -= d
                n[main["index"]] += d
        if n[main["index"]] < MIN_STAGE_LAYERS:
            warnings.append(f"the model has too few layers for this many stages: Main keeps {n[main['index']]}")
    else:                                                       # Main over-weight: hand layers to stages with time and room
        while n[main["index"]] > MIN_STAGE_LAYERS:
            room = [i for i in stages if n[i] < caps.get(i, n_layers)]
            if not room:
                break
            i = max(room, key=rate)
            if (n[i] + 1) / rate(i) >= (n[main["index"]] - 1) / rate(main["index"]):
                break
            n[i] += 1
            n[main["index"]] -= 1

    # ---- 2b. the x1 prefill budget: above X1_RESIDENT_LAYERS_SAFE whole layers on ALL slow-link dividers together,
    #      long-prompt prefill re-streams about 1 GiB per layer per chunk over the x1 link (measured: the constant) ----
    x1 = [i for i in stages if kind[i] == "divider" and _link(by_idx[i]) is not None and _link(by_idx[i]) <= slow_link_gbps]
    x1_total = sum(n[i] for i in x1)
    if max_x1_layers is not None:
        while x1_total > max(max_x1_layers, MIN_DIVIDER_LAYERS * len(x1)):
            k = max(x1, key=lambda i: n[i])
            if n[k] <= MIN_DIVIDER_LAYERS:
                break
            n[k] -= 1
            n[main["index"]] += 1
            x1_total -= 1
        if x1_total > max_x1_layers:
            warnings.append(f"the x1 dividers cannot drop below {x1_total} whole layers with {len(x1)} of them")
    elif x1_total > safe_x1_layers:
        warnings.append(f"the x1 dividers hold {x1_total} whole layers in total: past {safe_x1_layers} long-prompt prefill re-streams"
                        f" about 1 GiB per layer per chunk over the x1 link (637 -> 243 -> 142 tok/s at 11/13/15 on the measured rig),"
                        " while decode keeps improving - ask for max_x1_layers if long prompts matter more than decode speed")

    # ---- 3. the lead: a healthy link whenever one exists, else the slow stage with the most spare ----
    def spare(i: int, lead: bool = False) -> int:
        return _vram_bytes(by_idx[i]) - _fixed(kind[i], lead) - n[i] * (_per_layer(model) + bpt * ctx)
    healthy = [i for i in stages if _link(by_idx[i]) is None or _link(by_idx[i]) > slow_link_gbps]
    lead = max(healthy or stages, key=lambda i: (spare(i, True), -i))
    if not healthy:
        warnings.append(f"every helper is on a slow link: card {lead} leads the pipeline - tolerated (the entry feed"
                        " is ~KB per token); keep the picture reader on the CPU and the draft on Main")
    while lead is not None and n[lead] > MIN_STAGE_LAYERS and spare(lead, True) < 0:
        n[lead] -= 1
        n[main["index"]] += 1
        warnings.append(f"the lead card {lead} gives back layers to fund its prompt buffers")

    # ---- 4. order: the lead, then helpers (healthy and fast first), then dividers, Main last ----
    helpers = sorted([i for i in stages if kind[i] == "helper" and i != lead],
                     key=lambda i: (-(_link(by_idx[i]) or 10 ** 6), -rate(i)))
    dividers = sorted([i for i in stages if kind[i] == "divider" and i != lead],
                      key=lambda i: (-(_link(by_idx[i]) or 10 ** 6), -rate(i)))
    order = ([lead] if lead is not None else []) + helpers + dividers + [main["index"]]
    run, split_at, stage_out = 0, [], []
    for pos, i in enumerate(order):
        if pos:
            split_at.append(run)
        stage_out.append({"card": i, "role": kind[i], "first_layer": run, "layers": n[i],
                          "resident_bytes": n[i] * _per_layer(model) if kind[i] == "divider" else 0,
                          "spare_bytes": spare(i, pos == 0)})
        run += n[i]
    if run != n_layers:
        warnings.append(f"the split covers {run} of {n_layers} layers; check the card budgets")

    # ---- 5. the context the tightest stage can hold (when the model gives a per-token KV cost) ----
    context = None
    if bpt > 0:
        per_card = {s["card"]: max(0, int((_vram_bytes(by_idx[s["card"]]) - _fixed(s["role"], s is stage_out[0])
                                           - s["resident_bytes"]) // (bpt * max(s["layers"], 1)))) for s in stage_out}
        context = {"tokens": min(per_card.values()), "per_card": per_card}
        if ctx and context["tokens"] < ctx:
            warnings.append(f"the tightest card holds {context['tokens']} context tokens; the wanted {ctx} needs a rebalance")

    layer_split = ",".join(str(k) for k in split_at)
    lines = ["divider plan:"]
    for s in stage_out:
        c = by_idx[s["card"]]
        row = f"  {c.get('name', 'card ' + str(s['card']))}: {s['role']} layers {s['first_layer']}-{s['first_layer'] + s['layers'] - 1}"
        if s["role"] == "divider":
            row += f" ({s['resident_bytes'] / GB:.1f} GiB resident)"
        lk = _link(c)
        lines.append(row + f", link {'%.1f GB/s' % lk if lk else 'healthy/unknown'}")
    lines.append(f"  layer_split: {layer_split}")
    lines.append(f"  context fits: {context['tokens']} tokens" if context else "  context: unchecked (no per-token KV cost given)")
    for w in warnings:
        lines.append(f"  note: {w}")
    return {"roles": {str(i): kind[i] for i in kind}, "order": order, "layer_split": layer_split,
            "stages": stage_out, "context": context, "warnings": warnings, "plan_text": "\n".join(lines)}


# ---- the two I/O reads a live planner call needs (siblings of gpu_roles.nvidia_speeds: best effort, plain) ------------------------------

def model_from_pack(pack_dir) -> Optional[dict]:
    """{"n_layers", "expert_bytes_per_layer": [per layer]} from a native pack's native_experts.txt
    (`layer ... offset blob ... (n_expert N, total T; ...)`); None when the pack says nothing."""
    from pathlib import Path
    p = Path(pack_dir) / "native_experts.txt"
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    total = None
    rows: list[tuple[int, int]] = []
    for ln in lines:
        if ln.startswith("#"):
            head = ln.split("total", 1)
            if len(head) > 1:
                digits = ""
                for ch in head[1].replace(",", "").strip():
                    if ch.isdigit():
                        digits += ch
                    elif digits:
                        break
                total = int(digits) if digits else total
            continue
        parts = ln.split()
        if len(parts) >= 4 and parts[0].isdigit() and parts[3].isdigit():
            rows.append((int(parts[0]), int(parts[3])))
    if not rows or total is None:
        return None
    rows.sort()
    bytes_per = [rows[i + 1][1] - rows[i][1] for i in range(len(rows) - 1)] + [total - rows[-1][1]]
    return {"n_layers": len(rows), "expert_bytes_per_layer": bytes_per}


LANE_GBPS = {1: 1.0, 2: 2.0, 3: 4.0, 4: 8.0, 5: 16.0, 6: 32.0}    # theoretical PCIe GB/s per lane by generation


def nvidia_link_gbps(timeout: float = 15.0) -> dict[int, float]:
    """{nvidia-smi number: host->device GB/s} from the CURRENT link width and generation (an x1 riser reads ~1 lane);
    {} when nvidia-smi cannot say. The engine's own probe measures the real rate at start; this is the pre-start estimate."""
    import subprocess
    import sys
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,pcie.link.width.current,pcie.link.gen.current",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=timeout)
        out: dict[int, float] = {}
        for line in r.stdout.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 3 and all(p.isdigit() for p in parts[:3]):
                out[int(parts[0])] = int(parts[1]) * LANE_GBPS.get(int(parts[2]), 4.0)
        return out
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}
