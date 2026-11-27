"""Test a change to TH17 reward timing on the same four-game original PPO trajectories."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil

import numpy as np
from scripts.checkpoint_storage import digest
from scripts.verify_th17_original_credit import inspect


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','protocol','out'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args()
    cfg=json.loads(a.protocol.read_bytes())
    original=a.root/'corrected/credits.json'
    if digest(original)!=cfg['source_normalized_manifest_sha256']:raise ValueError('Frozen original PPO sample changed')
    rows=json.loads(original.read_bytes())
    gamma=cfg['ppo']['gamma']
    scale=cfg['th17_bomb_reserve_weight']
    if not 0<gamma<=1 or not 0<scale<1:raise ValueError('Invalid shaping parameters')
    a.out.mkdir(exist_ok=True,parents=True)
    audit=[]
    for row in rows:
        source=original.parent/row['file']
        if digest(source)!=row['sha256']:raise ValueError('Frozen original packed trace changed')
        target=a.out/row['file']
        if row['title']!=17:
            if not target.exists():
                try:os.link(source,target)
                except OSError:shutil.copy2(source,target)
            if digest(target)!=row['sha256']:raise ValueError('Control game changed')
            continue
        folder=a.root/'native17'/f"train-four-sft-{row['seed']}"
        rf=folder/'receipt.json'
        r=json.loads(rf.read_bytes())
        if digest(rf)!=row['source_log_sha256']:raise ValueError('Unmatched original stock receipt')
        inspect(rf,r,row['difficulty'])
        with (folder/'decisions.tsv').open(encoding='utf8') as stream:steps=list(csv.DictReader(stream,delimiter='\t'))
        with np.load(source,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        spares=np.array([int(s['spares_before']) for s in steps]);bombs=np.array([int(s['bombs_before']) for s in steps])
        actions=np.array([[int(s[k]) for k in ('move','focus','shoot','bomb')] for s in steps])
        if (len(steps)!=r['steps'] or np.min(spares)<0 or np.min(bombs)<0 or (spares[0],bombs[0])!=(2,3)
                or not np.array_equal(actions,arrays['action'])):raise ValueError('Unpaired native states or actions')
        phi=scale*(3*spares+bombs)
        delta=gamma*np.r_[phi[1:],0.]-phi
        residual=float(np.dot(gamma**np.arange(len(phi)),delta)+phi[0])
        if abs(residual)>1e-7:raise ValueError('Discounted shaping does not telescope across the complete credit')
        with (folder/'native.tsv').open(encoding='utf8') as stream:
            native=[s for s in csv.DictReader(stream,delimiter='\t') if int(s['stage'])>=1]
        variation={k:sorted({int(s[k]) for s in native}) for k in ('life_pieces','bomb_pieces')}
        arrays['original_reward']=arrays['reward'].copy()
        arrays['th17_stock_shaping']=delta.astype(np.float32)
        arrays['reward']=arrays['reward'].reshape(-1)+arrays['th17_stock_shaping']
        np.savez_compressed(target,**arrays)
        audit.append(dict(seed=row['seed'],decisions=len(steps),start_potential=float(phi[0]),discounted_shaping_total=-float(phi[0])+residual,
                          telescoping_residual=residual,mean_abs_increment=float(np.abs(delta).mean()),raw_fragment_values=variation,
                          source_sha256=row['sha256'],decision_state_sha256=r['decisions_sha256']))
        row['sha256']=digest(target)
    if len(audit)!=8:raise ValueError('Eight original TH17 credits required')
    (a.out/'credits.json').write_text(json.dumps(rows,indent=2)+'\n')
    (a.out/'reward-audit.json').write_text(json.dumps({'th17':audit,'other_titles':'Bit-identical in the normalized input to the learner.'},indent=2)+'\n')


if __name__=='__main__':main()
