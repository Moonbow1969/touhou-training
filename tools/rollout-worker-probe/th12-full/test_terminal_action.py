"""Regression for an original stock transition reported twice at the final decision tick."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from covered_events import final_input_after_loss,terminal_consumption

class TerminalBoundary(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory()
        self.raw=Path(self.folder.name)
        (self.raw/'coverage-decisions.tsv').write_text('step\tvisual\tused\tstage\tobserved_tick\tspares\n0\tplay\tplay\t1\t2196\t0\n1\tplay\tplay\t1\t2200\t0\n')
        self.live=np.array([(51478,1,2200,0,33),(51482,1,2200,-1,33),(51534,1,2201,-1,64)],dtype=[(n,'i8') for n in ('ms','stage','tick','spares','keys')])
        self.dec=np.array([(2196,2196,2200,40),(2200,2200,2204,1)],dtype=[(n,'i8') for n in ('pre_tick','apply_tick','post_tick','requested_x11')])
    def tearDown(self):self.folder.cleanup()
    def verify(self):return final_input_after_loss(self.raw,self.live,self.dec,1)
    def test_consumed_after_terminal(self):
        r=self.verify()
        self.assertEqual((r['terminal_credit_action'],r['first_excluded_action'],r['terminal_tick'],r['first_consumption_tick']),(0,1,2200,2201))
    def test_identical_inputs_cannot_identify_owner(self):
        self.dec['requested_x11'][-1]=40
        with self.assertRaises(ValueError):self.verify()
    def test_disjoint_interval_fails(self):
        self.dec['post_tick'][0]=2199
        with self.assertRaises(ValueError):self.verify()
    def test_new_keys_at_death_are_ambiguous(self):
        self.live['keys'][1]=64
        with self.assertRaises(ValueError):self.verify()
    def test_unobserved_postterminal_input_fails(self):
        self.live['keys'][2]=33
        with self.assertRaises(ValueError):self.verify()
    def test_not_live_at_new_camera_fails(self):
        (self.raw/'coverage-decisions.tsv').write_text('1\tplay\tplay\t1\t2200\t-1\n')
        with self.assertRaises(ValueError):self.verify()

class NativeConsumption(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory();self.raw=Path(self.folder.name)
        (self.raw/'coverage-decisions.tsv').write_text('0\tplay\tplay\t1\t2450\t0\n1\tplay\tplay\t1\t2455\t0\n')
        self.live=np.array([(50692,1,2455,0,65),(50737,1,2456,-1,65),(50753,1,2457,-1,97),(50769,1,2458,-1,97)],dtype=[(n,'i8') for n in ('ms','stage','tick','spares','keys')])
        self.dec=np.array([(1,2450,1,2450,2455,33),(1,2455,1,2455,2458,41)],dtype=[(n,'i8') for n in ('pre_stage','pre_tick','apply_stage','apply_tick','post_tick','requested_x11')])
    def tearDown(self):self.folder.cleanup()
    def check(self,death=1):return terminal_consumption(self.raw,self.live,self.dec,death)
    def test_old_command_still_consumed_after_nominal_end(self):
        r=self.check();self.assertEqual(r['live_actions'],1)
        self.assertEqual(r['previous_nominal_post_tick'],2455)
        self.assertEqual(r['prior_action_consumed_through_death_tick'],2456)
        self.assertEqual(r['first_consumption_tick'],2457)
    def test_new_input_on_actual_terminal_sample_has_explicit_order(self):
        self.live['keys'][1]=97
        self.assertEqual(self.check()['classification'],'new_mask_and_loss_first_observed_in_same_native_sample')
    def test_new_input_proven_while_alive(self):
        self.live['keys'][1]=97;self.live['spares'][1]=0
        self.assertEqual(self.check(2)['classification'],'new_mask_consumed_while_live')
    def test_identical_action_with_another_playable_tick(self):
        self.dec['requested_x11'][-1]=33;self.live['keys'][:]=65;self.live['spares'][1]=0
        self.assertEqual(self.check(2)['classification'],'held_mask_observed_on_a_subsequent_live_tick')
    def test_same_action_without_playable_tick_is_unknown(self):
        self.dec['requested_x11'][-1]=33;self.live['keys'][:]=65
        with self.assertRaises(ValueError):self.check()
    def test_native_tick_hole_is_unknown(self):
        self.live=self.live[[0,1,3]]
        with self.assertRaises(ValueError):self.check()
    def test_unrelated_consumed_keys_at_death_is_unknown(self):
        self.live['keys'][1]=1
        with self.assertRaises(ValueError):self.check()
if __name__=='__main__':unittest.main()
