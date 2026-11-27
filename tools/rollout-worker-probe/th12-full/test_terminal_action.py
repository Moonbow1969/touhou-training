"""Regression for an original stock transition reported twice at the final decision tick."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from covered_events import final_input_after_loss

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
if __name__=='__main__':unittest.main()
