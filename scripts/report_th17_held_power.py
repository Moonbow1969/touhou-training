"""Compare signed native Power trajectories between both arms of an original full-credit study."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from report_th17_power_pickups import inspect

def run(root):
    protocol=json.loads((root.parent/'camera-protocol.json').read_bytes())
    arms=(protocol['reference_arm'],protocol['candidate'])
    seeds=protocol['eval_seeds']['17']
    rows=[]
    for seed in seeds:
        pair=[]
        for arm in arms:
            folder=root/f'held-{arm}-{seed}'
            r=json.loads((folder/'receipt.json').read_bytes());detail=inspect(folder,'held')
            if detail['rank']!=int(protocol['rank_by_seed']['17'][str(seed)]) or r['continues']!=0:raise ValueError(folder)
            detail.update(arm=arm,lane=r['lane'],host=r['host'],trajectory_sha256=r['trajectory_sha256'],
                          unique_units_per_1000_ticks=1000*detail['unique_record_units']/max(1,detail['native_ticks']))
            if detail['nonrecord_units']<0:raise ValueError('Native high-water gain exceeds all positive pickup observations')
            pair.append(detail)
        if (pair[0]['lane'],pair[0]['host'])!=(pair[1]['lane'],pair[1]['host']):raise ValueError('Unmatched physical instance')
        rows.extend(pair)
    groups={}
    fields=('peak_displayed_power','unique_record_units','nonrecord_units','positive_power_units','unique_units_per_1000_ticks','native_ticks')
    for rank in (2,3):
        subset=[r for r in rows if r['rank']==rank]
        groups[rank]=dict(arms={a:{k:float(np.mean([r[k] for r in subset if r['arm']==a])) for k in fields} for a in arms},pairs={})
        for field in fields:
            deltas=defaultdict(list)
            for seed in sorted({r['seed'] for r in subset}):
                a,b=[r for r in subset if r['seed']==seed]
                deltas[a['lane']].append(b[field]-a[field])
            rng=np.random.default_rng(261126)
            draws=np.concatenate([rng.choice(v,(20000,len(v)),replace=True) for v in deltas.values()],axis=1).mean(axis=1)
            delta=np.concatenate(list(deltas.values()))
            groups[rank]['pairs'][field]=dict(n=len(delta),candidate_minus_reference=float(delta.mean()),paired_lane_bootstrap95=np.quantile(draws,[.025,.975]).tolist(),
                                             wins=int((delta>0).sum()),ties=int((delta==0).sum()),losses=int((delta<0).sum()))
    return dict(scope='Secondary original held endpoints fixed before this cohort. Peak Power uses playable native rows. Unique and non-record gain are in hundredths of displayed Power. Duration normalization is descriptive. Primary endpoints remain original stage and terminal resources.',by_rank=groups,credits=rows)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('native',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    result=run(a.native);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['by_rank'],indent=2))
