"""Run with python -m unittest scripts.test_th08_progress_reward from the repository root."""
import unittest

import numpy as np
from scripts.th08_progress_reward import GAMMA, progress_reward


class ProgressRewardTests(unittest.TestCase):
    def credit(self):
        return dict(stage=[1, 1, 1, 2, 2, 2], tick=[2, 7, 10, 3, 9, 11],
                    spares=[2, 1, 2, 2, 1, 0], continues=[0] * 6,
                    state_kind=["play"] * 6, spare_delta=[-1, 1, 0, -1, -1],
                    boundaries={2: (12, 1)}, stage_ticks={1: 20, 2: 40}, base_bootstrap=4.5)

    def assert_invariant(self, credit):
        raw = progress_reward(**credit, life_weight=0)
        shaped = progress_reward(**credit, life_weight=1)
        discounts = GAMMA ** np.arange(len(raw.reward))
        def total(a):
            return discounts @ a.reward + GAMMA ** len(discounts) * a.bootstrap
        self.assertAlmostEqual(total(shaped) - total(raw), -shaped.potential[0], places=12)
        # A shifted critic gives identical TD residuals and hence identical GAE
        # at every t, including an internal seam or a truncated final state.
        v = np.linspace(.5, 1.5, len(raw.reward))
        def td(a, v):
            return a.reward + GAMMA * np.r_[v[1:], a.bootstrap] - v
        np.testing.assert_allclose(td(shaped, v - shaped.potential[:-1]), td(raw, v), atol=1e-12)
        return raw, shaped

    def test_continuing_credit_truncated_in_two(self):
        raw, shaped = self.assert_invariant(self.credit())
        np.testing.assert_allclose(raw.progress, [.25, .15, .15, .15, .05])
        np.testing.assert_equal(raw.stage_bonus, [0, 0, 1, 0, 0])
        np.testing.assert_equal(shaped.potential, [3, 2, 3, 3, 2, 1])
        self.assertFalse(raw.terminated.any())
        self.assertEqual(shaped.bootstrap, 3.5)

    def test_stage_two_boundary_truncation_vs_objective(self):
        k = self.credit()
        for name in ("stage", "tick", "spares", "continues", "state_kind"):
            k[name] = k[name][:4]
        k["spare_delta"] = k["spare_delta"][:3]
        _, tr = self.assert_invariant(k)
        self.assertEqual(tr.bootstrap, 1.5)
        self.assertFalse(tr.terminated[-1])
        k.update(state_kind=["play"] * 3 + ["objective_terminal"], base_bootstrap=0)
        _, end = self.assert_invariant(k)
        self.assertEqual(end.potential[-1], 0)
        self.assertEqual(end.bootstrap, 0)
        self.assertTrue(end.terminated[-1])
        self.assertEqual(end.stage_bonus[-1], 1)

    def test_game_over_prompt_and_sentinel(self):
        for last in (0, -1):
            k = self.credit()
            k.update(spares=[2,1,2,1,0,last], spare_delta=[-1,1,-1,-1,0],
                     state_kind=["play"] * 5 + ["game_over"], base_bootstrap=None)
            _, a = self.assert_invariant(k)
            self.assertEqual(a.shaping[-1], -1)
            self.assertTrue(a.terminated[-1])

    def test_endpoint_labels_must_agree_with_stock_and_gameplay(self):
        for fields in ({"state_kind": ["play"] * 5 + ["game_over"]},
                       {"stage": [1,1,1,2,1,1]}, {"boundaries": {}},
                       {"tick": [2,7,6,3,9,11]}, {"boundaries": {2: (8,1)}},
                       {"spare_delta": [0,1,0,-1,-1]}, {"spares": [2,1,2,2,1,-1]},
                       {"state_kind": ["play", "game_over"] + ["play"]*4},
                       {"continues": [0,0,0,1,1,1]}, {"base_bootstrap": None},
                       {"tick": [2.,7.,10.,3.,9.,11.]}, {"stage_ticks": {1: 20}},
                       {"state_kind": ["play"]*5 + ["objective_terminal"]}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                progress_reward(**(self.credit() | fields), life_weight=1)

    def test_paused_clock_earns_zero_progress(self):
        k=self.credit()
        k["tick"][1]=k["tick"][0]
        a=progress_reward(**k, life_weight=0)
        self.assertEqual(a.reward[0],0.)


if __name__ == "__main__":
    unittest.main()
