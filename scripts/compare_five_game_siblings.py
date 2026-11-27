"""Pair the shared five-game update to the SFT and smaller PPO fits at identical original starts."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np

from scripts.checkpoint_storage import digest
from scripts.score_original_camera_run import records,verify
from scripts.score_native_stage_clock import summarize as clock
from scripts.verify_original_difficulty import verify as verify_rank


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--public',type=Path,required=True)
    args=parser.parse_args()
    data=args.data.resolve();root=data/'five-game-online'
    plan=json.loads((root/'camera-protocol.json').read_bytes())
    pending={10,11,13,17}
    while pending:
        for t in sorted(pending):
            output=root/'sibling-comparisons'/f'th{t}'
            scored=root/'evaluation'/f'th{t}'/'credits.json'
            if not scored.exists():continue
            source=data/('th11-original-ppo' if t==11 else 'four-game-merged-online')
            arm='th11-ppo' if t==11 else 'merged-ppo'
            fingerprint=digest(source/'policy/policy.ts')
            seeds=plan['eval_seeds'][str(t)]
            previous={}
            for title,f in records(source,True,True):
                if title!=t:continue
                r=json.loads(f.read_bytes())
                if r['seed'] not in seeds or r.get('policy_sha256')!=fingerprint:continue
                if t==13 and not r.get('ranking_photo'):continue
                if r['seed'] in previous:raise ValueError('Duplicate smaller PPO credit')
                previous[r['seed']]=(f,r)
            if set(previous)!=set(seeds):continue
            for name in ('endpoints','action-traces'):(output/name).mkdir(parents=True,exist_ok=True)
            rows=json.loads(scored.read_bytes())
            for seed,(f,r) in sorted(previous.items()):
                trace=f.parent/('trajectory.npz' if t==11 else r.get('behavior',r.get('trajectory',r.get('file','missing'))))
                row=verify(t,f,r,trace,arm,output)
                row.update(verify_rank(t,f,r,plan['rank_by_seed'][str(t)][str(seed)]))
                rows.append(row)
            for seed in seeds:
                triplet=[r for r in rows if r['seed']==seed]
                if len(triplet)!=3 or len({(r['host'],r['game_sha256'],r['difficulty']) for r in triplet})!=1:
                    raise ValueError(f'Unmatched original credits for {seed}')
            names=['merged-sft',arm,'five-ppo']
            result={}
            for k,label in ((2,'Hard'),(3,'Lunatic')):
                subset=[r for r in rows if r['difficulty']==k]
                by_arm={a:sorted([r for r in subset if r['arm']==a],key=lambda v:v['seed']) for a in names}
                result[label]={'n_pairs':len(subset)//3,'arms':{}}
                for a,group in by_arm.items():
                    if any(r['stage2_reached'] is None for r in group):raise ValueError('Unqualified stage outcome')
                    result[label]['arms'][a]=dict(stage2_hits=sum(r['stage2_reached'] for r in group),
                        mean_terminal_spares=np.mean([r['terminal_spares'] for r in group]),
                        mean_remaining_power_or_bombs=np.mean([r['terminal_power'] if t in (10,11) else
                        r.get('last_playable_bombs',r['terminal_bombs']) for r in group]))
                if t in (10,13,17):
                    result[label]['native_stage1_ticks']={a:clock(subset,t,(a,'five-ppo')) for a in names[:-1]}
            result.update(note='Exploratory same-seed comparison. Prior SFT and smaller PPO credits share the physical host and original build with the five-game candidate and predate it. TH12 has a changed controller and fresh seeds; it is excluded. The original game RNG is uncontrolled.',
                          policy_sha256={a:next(r['policy_sha256'] for r in rows if r['arm']==a) for a in names},
                          per_credit=[{k:r[k] for k in ('seed','arm','host','difficulty','game_sha256','original_trace_sha256',
                            'native_log_sha256','stage2_reached','terminal_spares','terminal_power','terminal_bombs','effective_bombs','latest_observed_stage1_native_tick')} for r in rows])
            (output/'paired.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
            dest=args.public/'five-game-online'/output.relative_to(root)
            dest.mkdir(parents=True,exist_ok=True)
            shutil.copy2(output/'paired.json',dest/'paired.json')
            print('Sealed three-way original comparison',t,flush=True)
            pending.remove(t)
        if pending:time.sleep(24)


if __name__=='__main__':
    main()
