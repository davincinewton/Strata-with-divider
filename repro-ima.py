#!/usr/bin/env python3
# Repro for the 6-divider fleet "illegal instruction" crash (Xid 13, PC 0xc81eb60).
# Sends a ~10k-token prompt (failure zone is 3-6k tokens) to the divider server.
import json, sys, time, urllib.request

def build_text(target_chars):
    base = ("The lighthouse keeper climbed the spiral stair at dawn, logging the swell height "
            "in his worn notebook. A gull wheeled over the gallery rail and the foghorn answered "
            "slowly across the water. He checked the logbook, the oil, the wick, and the barometer. "
            "Each entry recorded the hour, the bearing, the wind, and the state of the sea.\n\n")
    n = (target_chars // len(base)) + 1
    return (base * n)[:target_chars]

def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 45_000   # ~10.5k tokens
    text = build_text(target)
    body = json.dumps({
        "model": "default",
        "messages": [{"role": "user",
                      "content": text + "\n\nIn one short sentence: what is the keeper watching?"}],
        "max_tokens": 8,
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:8081/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    print(f"sending {len(text)} chars (~{len(text)//4.3:.0f} tokens) at {time.strftime('%H:%M:%S')}", flush=True)
    try:
        with urllib.request.urlopen(req, timeout=7200) as r:
            out = json.load(r)
        print(f"OK in {time.time()-t0:.1f}s usage={out.get('usage')}", flush=True)
    except Exception as e:
        print(f"ERROR after {time.time()-t0:.1f}s: {type(e).__name__} {e}", flush=True)

if __name__ == "__main__":
    main()