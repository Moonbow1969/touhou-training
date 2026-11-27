"""Pack pre-action camera frames and complete frozen behavior from registered native credits."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.checkpoint_storage import digest
from scripts.score_original_camera_run import records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','behavior','protocol','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--round2',action='store_true')
    a=p.parse_args();plan=json.loads(a.protocol.read_bytes())
    wanted={(int(t),s) for t,seeds in plan['train_seeds'].items() for s in seeds}
    fingerprint=digest(a.behavior);found={}
    for title,receipt in records(a.root,True,a.round2):
        r=json.loads(receipt.read_bytes());key=(title,r.get('seed'))
        if key not in wanted or r.get('policy_sha256')!=fingerprint:continue
        if key in found:raise ValueError(f'Duplicate training credit {key}')
        if r.get('continues',0) or not r.get('terminal',True):raise ValueError(f'Incomplete credit {key}')
        state=r.get('outcome',r.get('native_outcome','')).lower()
        if any(x in state for x in ('game_over','ranking','final_life')):outcome='game_over'
        elif any(x in state for x in ('ending','clear')):outcome='ending'
        else:raise ValueError(f'Unknown end state {state}')
        found[key]=(receipt,r,outcome)
    if set(found)!=wanted:raise ValueError(f'Missing registered training credits: {sorted(wanted-set(found))}')
    a.out.mkdir(parents=True,exist_ok=True);items=[]
    for (title,seed),(receipt,r,outcome) in sorted(found.items()):
        trace=receipt.parent/r.get('behavior',r.get('trajectory',r.get('file','missing')))
        trace_sha=r.get('behavior_sha256',r.get('trajectory_sha256',r.get('sha256')))
        if digest(trace)!=trace_sha:raise ValueError(f'Changed original behavior {trace}')
        with np.load(trace,allow_pickle=False) as z:
            arrays={k:z[k] for k in ('action','logp','value','reward','terminated')}
            frames=z['frames'] if 'frames' in z else None
        n=len(arrays['action'])
        if frames is None and title==10:
            with np.load(receipt.parent/'original-pixels.npz',allow_pickle=False) as z:frames=z['obs']
            if len(frames)==n+1:frames=frames[:-1]
        if frames is None and title==12:
            frames=np.fromfile(receipt.parent.parent/'raw'/r['episode']/'observations.u8',np.uint8).reshape(-1,112,96)
        if frames is None or frames.dtype!=np.uint8 or frames.shape!=(n,112,96):raise ValueError(f'Frame/action mismatch {trace}')
        done=arrays['terminated'].reshape(-1)
        if not n or not done[-1] or done[:-1].any():raise ValueError(f'Unverified terminal in {trace}')
        target=a.out/f'th{title}-{seed}.npz'
        np.savez_compressed(target,frames=frames,**arrays)
        items.append(dict(title=title,seed=seed,policy_sha256=fingerprint,role='train',outcome=outcome,
                          file=target.name,sha256=digest(target),source_log_sha256=digest(receipt),source_actions_sha256=trace_sha))
    (a.out/'credits.json').write_text(json.dumps(items,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
