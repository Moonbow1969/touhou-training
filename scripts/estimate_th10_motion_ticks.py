"""Estimate active native motion ticks from the original player's discrete movement increments."""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from scripts.collect_th10_full_action import save


def samples(path):
    with gzip.open(path,'rt') as f:rows=list(csv.DictReader(f))
    times=np.array([int(s['utc_ms']) for s in rows]);lives=np.array([int(s['lives']) for s in rows])
    xy=np.array([[float(s['x']),float(s['y'])] for s in rows])
    keys=np.array([int(s['native_input'])&(240|4|2) for s in rows])
    out=[]
    for i in range(1,len(rows)-2):
        key=keys[i];direction=key&240
        if direction not in (16,32,64,128) or key&2:continue
        if not np.all(keys[i-1:i+3]==key) or not np.all(lives[i-1:i+3]==lives[i]):continue
        points=xy[i-1:i+3]
        if (np.abs(points[:,0])>160).any() or (points[:,1]<30).any() or (points[:,1]>400).any():continue
        axis=0 if direction in (64,128) else 1
        sign=1 if direction in (128,32) else -1
        dxy=points[1:]-points[:-1]
        along=dxy[:,axis]*sign
        if (np.abs(dxy[:,1-axis])>.02).any() or (along<-.02).any() or max(along)<.2:continue
        dt=times[i+1]-times[i]
        if 0<dt<=100:out.append((int(bool(key&4)),round(float(along[1]),2),int(dt)))
    return out


def main(base,role='train'):
    manifest=json.loads((base/('ppo-receipts.json' if role=='train' else 'held-receipts.json')).read_bytes())
    assert len(manifest)==8 and all(r['role']==role for r in manifest)
    per_seed={};counts=[Counter(),Counter()]
    for r in manifest:
        d=base/f"{r['host']}-{r['seed']}"
        native=samples(d/'native.csv.gz');per_seed[r['seed']]=(d,native)
        for focus,v,dt in native:
            if v>0:counts[focus][v]+=1
    # Two frequency peaks represent one and two motion increments between the ~25ms samples.
    unit=[]
    for counts_by_focus in counts:
        peaks=sorted(v for v,n in counts_by_focus.most_common(2))
        assert abs(peaks[1]-2*peaks[0])<.03
        unit.append(peaks[0])
    records=[];all_steps=0;all_ms=0
    for seed,(d,data) in per_seed.items():
        ticks=0;dt=0;kept=0
        for focus,v,t in data:
            n=round(v/unit[focus])
            if 0<=n<=6 and abs(v-n*unit[focus])<=.04:
                ticks+=n;dt+=t;kept+=1
        with np.load(d/'original-pixels.npz') as p:control_ms=float(np.median(np.diff(p['monotonic_ns']))/1e6)
        rate=1000*ticks/dt if kept>=5 and dt>0 else None;all_steps+=ticks;all_ms+=dt
        records.append(dict(seed=seed,valid_sensor_pairs=kept,total_candidates=len(data),motion_increments=ticks,
                            sensor_time_ms=dt,motion_increments_per_second=rate,median_control_ms=control_ms,
                            estimated_motion_increments_per_decision=rate*control_ms/1000 if rate is not None else None))
    estimates=[r['estimated_motion_increments_per_decision'] for r in records if r['estimated_motion_increments_per_decision'] is not None]
    assert all_ms>0 and estimates
    result=dict(role=role,source='Eight separate original TH10 credits. The native sensor samples about every 25ms. Select four consecutive stable same-life cardinal direction/focus readings in the playfield interior. Isolate movement on the commanded axis. At least one of the three adjacent displacements is positive. The two most frequent displacements identify one and two motion increments. Include zero-increment pairs where neighboring readings are moving. A per-credit rate needs at least five qualified sensor pairs. This estimates native active motion pacing, not a direct engine-clock register.',
        displacement_units_per_increment={'unfocused':unit[0],'focused':unit[1]},
        observed_peak_counts_by_focus=[c.most_common(6) for c in counts],
        pooled_measured_motion_increments_per_second=1000*all_steps/all_ms,
        median_per_credit_motion_increments_per_decision=float(np.median(estimates)),
        native_sensor_time_ms=all_ms,ticks=all_steps,credits=records)
    save(base/'native-motion-cadence.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('base',type=Path)
    p.add_argument('--role',choices=['train','held'],default='train')
    args=p.parse_args();print(json.dumps(main(args.base,args.role),indent=2))
