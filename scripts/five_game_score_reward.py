"""A bounded native TH11 score increment on the same five-title on-policy training sample."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze
from scripts.train_th11_original_ppo import audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--public',type=Path,required=True)
    args=p.parse_args();data=args.data.resolve()
    joint=data/'five-game-online'
    root=data/'five-game-th11-score';root.mkdir(exist_ok=True)
    out=root/'training';out.mkdir(exist_ok=True)
    parent=data/'specialist-vs-shared/merged-equal'
    plan=json.loads((joint/'camera-protocol.json').read_bytes())
    stock=data/'th11-original-ppo'
    one=json.loads((stock/'camera-protocol.json').read_bytes())
    held=list(range(1471151,1471167))
    plan['rank_by_seed']={k:{str(s):2 if s%2 else 3 for s in seeds} for k,seeds in plan['train_seeds'].items()}
    plan['rank_by_seed']['11'].update({str(s):2 if s%2 else 3 for s in held})
    plan.update(name='Bounded original score reward on TH11 inside the five-title PPO update',
                reference_arm='five-ppo',candidate='five-score-ppo',eval_seeds={'11':held},
                reference_evaluation_sha256=digest(joint/'policy/policy.ts'),
                native_executable_sha256=one['native_executable_sha256'],
                design='Same original human SFT, 48 physical full credits, observations, log probabilities, action labels, initial critic, ten equally weighted strata, four epochs and optimization seed as the five-game reference. Change only the reward supplied on the 16 TH11 training credits.',
                reward='Use the original native points P completed during each action, with F(P)=P/(P+233975). Add 0.30 times the increment in F to the original stock-change reward. The scale 233975 is the median fatal-frame score of the 16 TRAIN credits. The entire score bonus is bounded by 0.30 per credit, below one spare life. No score observation enters the policy. This changes the learning objective.',
                evaluation='Fresh physical two-arm whole-credit TH11 evaluation only, eight pairs per rank. Alternate order within rank on the same physical lane. Primary endpoint Stage 2, with spare lives and Power at entry and terminal. Report native score at final loss, settled score, effective casts, and score at first life loss. This exploratory follow-up was motivated by prior development results. No held seed or image enters the fit.')
    protocol=root/'camera-protocol.json';freeze(protocol,plan)
    old=json.loads((joint/'training/credits.json').read_bytes())
    stats=[];rows=[]
    for record in old:
        row=dict(record)
        source=joint/'training'/row['file'];target=out/row['file']
        if digest(source)!=row['sha256']:raise ValueError('Changed original training input')
        if row['title']!=11:
            if not target.exists():os.link(source,target)
            if digest(target)!=row['sha256']:raise ValueError('Control game changed')
        else:
            seed=row['seed'];folder=stock/'native11'/f'train-merged-sft-{seed}'
            audit(folder,'train',seed,plan['behavior_sha256'],plan)
            with np.load(source,allow_pickle=False) as z:
                arrays={k:z[k] for k in z}
            native=arrays['native'].astype(np.int64)
            points=native[:,1]*10
            if points[0]!=0 or np.any(np.diff(points)<0):raise ValueError('Nonmonotone original points')
            with (folder/'behavior.csv').open(encoding='utf8') as stream:
                decisions=list(csv.DictReader(stream))
            origin=int(decisions[0]['before_ms'])
            ends=np.asarray([(int(r['after_ms'])-origin)%(2**32) for r in decisions])
            times=(native[:,0]-origin)%(2**32)
            changes=np.flatnonzero(np.diff(points)>0)+1
            index=np.searchsorted(ends,times[changes])
            increments=np.zeros(len(ends),dtype=np.int64)
            np.add.at(increments,index,points[changes]-points[changes-1])
            cumulative=increments.cumsum()
            if cumulative[-1]!=points[-1]:raise ValueError('Lost native score event')
            potential=cumulative/(cumulative+233975.)
            bonus=.30*np.diff(potential,prepend=0)
            if bonus.min()<0 or not 0<=bonus.sum()<.30:raise ValueError('Invalid score reward')
            original=arrays['reward'].reshape(-1)
            arrays['reward_stock_only']=original
            arrays['native_score_increments']=increments
            arrays['reward']=(original+bonus).astype(np.float32).reshape(arrays['reward'].shape)
            if not target.exists():np.savez_compressed(target,**arrays)
            with np.load(target,allow_pickle=False) as z:
                if set(z.files)!=set(arrays):raise ValueError('Changed reward archive schema')
                for key,value in arrays.items():np.testing.assert_array_equal(z[key],value)
            row['original_stock_only_sha256']=row['sha256'];row['sha256']=digest(target)
            stats.append(dict(seed=seed,rank=row['difficulty'],decisions=len(ends),positive_score_decisions=int((bonus>0).sum()),
                              original_return=float(original.sum()),bonus_return=float(bonus.sum()),fatal_score=int(points[-1]),
                              native_evidence_sha256=digest(folder/'native.bin'),behavior_sha256=digest(folder/'behavior.csv')))
        rows.append(row)
    if len(stats)!=16 or np.median([r['fatal_score'] for r in stats])!=233975:
        raise ValueError('Changed training population or predeclared score scale')
    freeze(out/'credits.json',rows);freeze(root/'score-reward-audit.json',stats)
    if not (root/'policy/report.json').exists():
        subprocess.run([sys.executable,'-u','-m','scripts.train_full_resolution_camera_ppo','--manifest',out/'credits.json',
                        '--protocol',protocol,'--behavior',parent/'policy.ts','--sft',parent/'camera-policy.pt',
                        '--out',root/'policy'],check=True)
    report=json.loads((root/'policy/report.json').read_bytes())
    if report['steps']!=18223 or report['protocol_sha256']!=digest(protocol) or report['policy_sha256']!=digest(root/'policy/policy.ts'):
        raise ValueError('Invalid controlled fit')
    for file in [protocol,out/'credits.json',root/'score-reward-audit.json',*(root/'policy').iterdir()]:
        dest=args.public/root.name/file.relative_to(root);dest.parent.mkdir(exist_ok=True,parents=True)
        shutil.copy2(file,dest)
    print('Fitted score reward sibling',report['policy_sha256'],'on',sum(r['positive_score_decisions'] for r in stats),'score-bearing actions',flush=True)


if __name__=='__main__':
    main()
