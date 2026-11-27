"""Describe spirit-strike probabilities around original life-counter losses on training credits."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from scripts.collect_th10_full_action import save


def main(directory):
    manifest=json.loads((directory/'ppo-receipts.json').read_bytes())
    assert len(manifest)==8 and all(r['role']=='train' for r in manifest)
    groups={key:[] for key in ('all_play','power_eligible','eligible_before_life_loss','eligible_elsewhere')}
    credits=[];lags=[]
    for r in manifest:
        folder=directory/f"momentum-{r['seed']}"
        report=json.loads((folder/'report.json').read_bytes())
        with gzip.open(folder/'native.csv.gz','rt') as f:native=list(csv.DictReader(f))
        with np.load(folder/'behavior.npz') as b:
            ts=b['action_utc_ms'];logits=b['camera_logits'][:,13:15].astype(np.float64)
            probability=1/(1+np.exp(logits[:,0]-logits[:,1]))
            bomb_request=b['action'][:,3].astype(bool)
        nt=np.asarray([int(row['utc_ms']) for row in native]);lives=np.asarray([int(row['lives']) for row in native])
        power=np.asarray([int(row['power_units']) for row in native])
        idx=np.searchsorted(nt,ts,side='right')-1
        lag=ts-nt[np.maximum(idx,0)]
        valid=(idx>=0)&(lag<=100)
        lags.extend(lag[valid].tolist())
        eligible=valid&(power[np.maximum(idx,0)]>=20)
        hits=nt[1:][np.diff(lives)<0]
        dt=hits[None,:]-ts[:,None]
        before=((dt>0)&(dt<=500)).any(1)
        for key,mask in [('all_play',valid),('power_eligible',eligible),('eligible_before_life_loss',eligible&before),('eligible_elsewhere',eligible&~before)]:
            groups[key].append(np.stack([probability[mask],bomb_request[mask]],axis=1))
        credits.append(dict(seed=r['seed'],decisions=len(ts),valid_preceding_native_inventory=int(valid.sum()),
            power_eligible_decisions=int(eligible.sum()),life_losses=len(hits),confirmed_spirit_strikes=len(report['successful_bombs']),
            bomb_requested_decisions=int(bomb_request.sum())))
    result=dict(credits=credits,window_ms=500,max_inventory_sample_lag_ms=100,observed_lag_ms_p95=float(np.quantile(lags,.95)),
        scope='Eight complete training credits under camera SFT. Probability is computed from acted logits. Eligible means the last original inventory at action application has at least 1.00 Power. A subsequent counter loss is observational. Successful strikes can prevent losses, and counter timing can follow collision.',
        groups={})
    for key,rows in groups.items():
        matrix=np.concatenate(rows);result['groups'][key]=dict(n=len(matrix),p_bomb_mean=float(matrix[:,0].mean()) if len(matrix) else None,
                  requests=int(matrix[:,1].sum()),request_fraction=float(matrix[:,1].mean()) if len(matrix) else None)
    save(directory/'spirit-strike-timing.json',result)
    print(json.dumps(result['groups'],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    main(parser.parse_args().directory)
