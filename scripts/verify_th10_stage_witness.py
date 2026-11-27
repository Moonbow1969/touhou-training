"""Exercise the production native stage-reward path on the genuine two-register transition."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

from scripts.collect_th10_full_action import native_stage_rewards,save


def verify(folder):
    calibration=json.loads((folder/'sensor-repair.json').read_bytes())['second_witness']
    trace=folder/calibration['trace']
    assert hashlib.sha256(trace.read_bytes()).hexdigest()==calibration['checksums']['diagnostic.csv']
    with trace.open() as f:original=list(csv.DictReader(f))
    clocks=np.array([int(r['utc_ms']) for r in original])
    with np.load(folder/calibration['camera']) as z:camera=z['utc_ms']
    results={}
    for name in ('legacy_stage','candidate_stage'):
        native=[dict(stage=r[name],lives=r['spares'],power_units=r['power_units'],utc_ms=r['utc_ms']) for r in original]
        reward,entries=native_stage_rewards(native,clocks)
        first_camera,_=native_stage_rewards(native,camera)
        results[name]=dict(native_input_intervals=len(clocks),total_stage_reward=float(reward.sum()),
            rewarded_native_intervals=np.flatnonzero(reward).tolist(),rewarded_policy_camera_intervals=np.flatnonzero(first_camera).tolist(),
            stage_entries=entries)
    old,new=results.values()
    assert old['total_stage_reward']==0 and new['total_stage_reward']==2
    assert len(new['rewarded_native_intervals'])==len(new['rewarded_policy_camera_intervals'])==1
    assert new['stage_entries'][1]['lives']==1 and new['stage_entries'][1]['power']==.1
    save(folder/'production-stage-reward-witness.json',dict(measurement='Production stage reward reconstruction on the real isolated original-game calibration; not a learned policy or training example.',sensor=results))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
    verify(p.parse_args().folder)
