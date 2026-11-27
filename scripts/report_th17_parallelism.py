"""Summarize actual simultaneous native TH17 progress on the shared Gamer monotonic clock."""
from pathlib import Path
import argparse,json
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/rollout-worker-probe/th17-full'))
from timeline import progressed_ticks

def report(root):
    credits=[]
    for p in sorted(root.glob('*-*-*/receipt.json')):
        r=json.loads(p.read_bytes())
        t=np.genfromtxt(p.with_name('decisions.tsv'),names=True,delimiter='\t')
        credits.append((r,np.r_[t['onset_ns'][0],t['observed_ns']]/1e9,progressed_ticks(np.r_[t['stage'][0],t['after_stage']],np.r_[t['tick'][0],t['after_tick']])))
    cell={};wall={}
    if not credits:return {}
    # Count only seconds fully covered by every included active credit.
    for start in np.arange(np.ceil(min(ts[0] for _,ts,_ in credits)),np.floor(max(ts[-1] for _,ts,_ in credits))):
        spans=[(r,ts,ticks) for r,ts,ticks in credits if ts[0]<=start<start+1<=ts[-1]]
        if not spans:continue
        # Drop windows where a new episode enters partway or another exits early.
        if any(start<ts[0]<start+1 or start<ts[-1]<start+1 for _,ts,_ in credits):continue
        count=len(spans)
        keys={r['lane'] for r,_,_ in spans}
        if len(keys)!=count:raise ValueError('Two original credits appear concurrent in one lane')
        cell.setdefault(count,[]).append(sum(np.interp(start+1,ts,ticks)-np.interp(start,ts,ticks) for _,ts,ticks in spans))
        wall.setdefault(count,[]).append(sum(np.searchsorted(ts,start+1)-np.searchsorted(ts,start) for _,ts,ticks in spans))
    groups={}
    for count,values in cell.items():
        groups[count]=dict(overlap_seconds=len(values),mean_aggregate_native_ticks_per_second=float(np.mean(values)),
            median_aggregate_native_ticks_per_second=float(np.median(values)),p10_p90_aggregate_native_ticks_per_second=np.percentile(values,[10,90]).tolist(),
            mean_aggregate_four_head_decisions_per_second=float(np.mean(wall[count])))
    return dict(credits=len(credits),original_cpu_quota_per_lane=2,host='Gamer-WSL',scope='Exact on-policy native ticks and applied four-head decisions per real second. Integer one-second bins on the shared monotonic clock; partial start/end windows excluded.',groups=groups)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    o=report(a.root);a.out.write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))

