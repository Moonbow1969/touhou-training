"""Synthetic regression only: music timer resets both inside a stage and at native Stage 2."""
from pathlib import Path
import sys
import unittest
import numpy as np
here=Path(__file__).resolve().parent
sys.path.insert(0,str(here.parents[2]))
from scripts.verify_th17_original_credit import verify_stage_chronology
from receipt import state_events
from timeline import progressed_ticks, initialized_stock_rows

class StageClockTest(unittest.TestCase):
    def setUp(self):
        self.rows=np.genfromtxt(here/'synthetic-music-wrap.tsv',names=True,delimiter='\t',dtype=None,encoding='utf8')
        self.decisions=np.array([(1,12),(1,16),(1,20),(2,1),(2,5),(2,9),(2,16)],dtype=[('after_stage','i4'),('after_tick','i4')])
    def test_music_wrap_is_valid_and_stock_uses_stage_tick(self):
        verify_stage_chronology(self.rows)
        self.assertGreater(np.count_nonzero(np.diff(self.rows['total_tick'])<0),1)
        delta=progressed_ticks(self.rows['stage'],self.rows['tick'])
        np.testing.assert_array_equal(delta,[0,6,12,12,15,20,28])
        reward,events,stage=state_events(self.rows,self.decisions)
        self.assertEqual(stage[2],dict(step=3,tick=0,spares=1,bombs=3))
        self.assertEqual([e['step'] for e in events if e['kind']=='spent_bombs'],[1,4])
        self.assertEqual([e['step'] for e in events if e['kind']=='life'],[2,5,6])
        np.testing.assert_allclose(reward.sum(0),[-3,-.06,2,0])
    def test_per_stage_clock_rollback_is_rejected(self):
        self.rows['tick'][2]=4
        with self.assertRaises(ValueError):verify_stage_chronology(self.rows)
        with self.assertRaises(ValueError):progressed_ticks(self.rows['stage'],self.rows['tick'])
    def test_skipping_native_stage_is_rejected(self):
        self.rows['stage'][3:]=3
        with self.assertRaises(ValueError):verify_stage_chronology(self.rows)
        with self.assertRaises(ValueError):progressed_ticks(self.rows['stage'],self.rows['tick'])
    def test_only_original_zero_initializer_precedes_stock(self):
        dtype=[(k,'i4') for k in ('stage','start_stage','tick','power','spares','bombs','continues','score')]
        rows=np.array([(1,1,0,0,0,0,0,0),(1,1,0,0,2,3,0,0),(1,1,1,100,2,3,0,0)],dtype=dtype)
        trimmed,count=initialized_stock_rows(rows,2)
        self.assertEqual((count,len(trimmed),trimmed[0]['spares'],trimmed[0]['bombs']),(1,2,2,3))
        rows['tick'][0]=1
        self.assertEqual(initialized_stock_rows(rows,2)[1],0)
if __name__=='__main__':unittest.main()
