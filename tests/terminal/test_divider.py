"""strata_control/divider.py: the divider planner (whole layers on a slow-link card, link-first order, three budgets)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from strata_control import divider as D

ROOT = Path(__file__).resolve().parents[2]
V = json.loads((ROOT / "tests" / "data" / "divider_vectors.json").read_text(encoding="utf-8"))


class Vectors(unittest.TestCase):
    def test_plan(self):
        for c in V["plan"]:
            with self.subTest(c["why"]):
                r = D.plan(V["hw"][c["rig"]], V["model"][c["model"]], c.get("roles"))
                want = c["expect"]
                if want is None:
                    self.assertIsNone(r)
                    continue
                self.assertIsNotNone(r)
                self.assertEqual(r["order"], want["order"])
                self.assertEqual(r["layer_split"], want["layer_split"])
                if "roles" in want:
                    self.assertEqual(r["roles"], want["roles"])
                if "context_tokens" in want:
                    self.assertEqual(r["context"]["tokens"], want["context_tokens"])
                for w in want.get("warn_any", []):
                    self.assertTrue(any(w in x for x in r["warnings"]), f"missing warning {w!r} in {r['warnings']}")


class Rules(unittest.TestCase):
    """The rules the vectors cannot show on their own."""

    def test_main_is_last_and_dividers_never_last(self):
        r = D.plan(V["hw"]["rig9"], V["model"]["q6xl"])
        self.assertEqual(r["order"][-1], 0)                      # Main is the last stage by architecture
        self.assertEqual(r["stages"][-1]["role"], "main")
        self.assertTrue(all(s["role"] != "divider" for s in r["stages"][-1:]))

    def test_cover_and_minimums(self):
        r = D.plan(V["hw"]["rig9"], V["model"]["q6xl"])
        self.assertEqual(sum(s["layers"] for s in r["stages"]), V["model"]["q6xl"]["n_layers"])
        self.assertTrue(all(s["layers"] >= D.MIN_STAGE_LAYERS for s in r["stages"]))
        for s in r["stages"]:
            if s["role"] == "divider":
                self.assertGreaterEqual(s["layers"], D.MIN_DIVIDER_LAYERS)
                self.assertGreater(s["resident_bytes"], 0)

    def test_layer_split_matches_order(self):
        r = D.plan(V["hw"]["mixed"], V["model"]["small"])
        split = [int(k) for k in r["layer_split"].split(",")]
        self.assertEqual(split, [s["first_layer"] for s in r["stages"][1:]])

    def test_explicit_main_obeys_the_person(self):
        r = D.plan(V["hw"]["mixed"], V["model"]["small"], {0: "main"})
        self.assertEqual(r["order"][-1], 0)
        self.assertEqual(r["roles"]["0"], "main")

    def test_speedless_cards_still_plan(self):
        cards = [{"index": 0, "name": "A", "vram_gb": 24, "link_gbps": 20.0},
                 {"index": 1, "name": "B", "vram_gb": 24, "link_gbps": 0.4}]
        r = D.plan(cards, V["model"]["q6xl"])
        self.assertEqual(r["roles"]["1"], "divider")
        self.assertTrue(any("no speed reading" in w for w in r["warnings"]))

    def test_usable_false_and_igpu_never_stage(self):
        cards = [{"index": 0, "name": "A", "vram_gb": 24, "speed": 200, "link_gbps": 20.0},
                 {"index": 1, "name": "B", "vram_gb": 24, "speed": 150, "usable": False, "link_gbps": 30.0},
                 {"index": 2, "name": "iGPU", "vram_gb": 0, "speed": 10, "integrated": True}]
        r = D.plan(cards, V["model"]["q6xl"])
        self.assertEqual(r["order"], [0])
        self.assertEqual(r["roles"]["1"], "off")
        self.assertNotIn("2", r["roles"])


class LaunchWiring(unittest.TestCase):
    """The launch side: divider_cards sizes a real mirror, and apply_cards merges its explicit split and environment."""

    def test_apply_cards_merges_env_add_and_keeps_the_explicit_split(self):
        from strata_cli.launch.runner import apply_cards
        cfg = {"env": {"STRATA_MMQ_KQ": "1"}, "layer_split": "auto", "gpu": 9}
        apply_cards(cfg, {"gpu": [3, 1, 0], "gpu_order": "as_given", "gpu_roles": {"0": "main", "1": "divider", "3": "divider"},
                          "gpu_cards": {}, "layer_split": "7,15", "env_add": {"STRATA_STAGE_TRIM": "1"}})
        self.assertEqual(cfg["gpu"], [3, 1, 0])
        self.assertEqual(cfg["layer_split"], "7,15")
        self.assertEqual(cfg["env"], {"STRATA_MMQ_KQ": "1", "STRATA_STAGE_TRIM": "1"})
        self.assertNotIn("env_add", cfg)

    def test_divider_cards_without_a_native_pack_falls_back_plain(self):
        from strata_cli.launch.runner import divider_cards
        from strata_control import gpu_roles
        cards = [{"index": 0, "name": "A", "vram_gb": 24, "speed": 200}, {"index": 1, "name": "B", "vram_gb": 22, "speed": 105}]
        roles = {0: "main", 1: "divider"}
        m = divider_cards(gpu_roles.mirror(roles, cards), roles, cards, None)
        self.assertIn("native_experts.txt", m["divider_plan"])
        self.assertEqual(m["gpu"], [1, 0])                     # the plain mirror stands: helper before Main
        self.assertNotIn("env_add", m)


if __name__ == "__main__":
    unittest.main()
