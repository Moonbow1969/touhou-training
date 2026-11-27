"""Recount high-difficulty optimizer exposures from original credits, before paired evaluation."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

from scripts.checkpoint_storage import digest
from scripts.score_original_camera_run import records, verified_th10_entry, verified_th12_stage
from scripts.verify_original_difficulty import verify


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    plan=json.loads((a.root/'camera-protocol.json').read_bytes())
    learner=json.loads((a.root/'policy/report.json').read_bytes())
    wanted={(int(t),s) for t,seeds in plan['train_seeds'].items() for s in seeds}
    used={(x['title'],x['seed']):x for x in learner['credits']}
    if set(used)!=wanted or learner['policy_sha256']!=digest(a.root/'policy/policy.ts'):raise ValueError('PPO checkpoint or source set differs')
    rows=defaultdict(list)
    seen=set()
    for title,f in records(a.root,True,True):
        r=json.loads(f.read_bytes())
        key=title,r.get('seed')
        if key not in wanted:continue
        if key in seen or r['policy_sha256']!=learner['native_behavior_sha256']:raise ValueError('Duplicate or inconsistent original credit')
        seen.add(key)
        rank=plan['rank_by_seed'][str(title)][str(key[1])]
        evidence=verify(title,f,r,rank)
        if title==10:
            stage=verified_th10_entry(r,f.parent)[0]
            spares,bombs,power=r['final_lives'],None,r['final_power']
        elif title==12:
            stage=verified_th12_stage(f.parent.parent/'raw'/r['episode'])[0]
            spares,bombs,power=r['remaining_spare_lives'],r['remaining_bombs'],None
        elif title==13:
            stage=r['stage2_reached']
            if not (f.parent/r['ranking_photo']).is_file():raise ValueError('Missing original ranking image')
            spares,bombs,power=r['remaining_lives'],r['remaining_bombs'],None
        elif title==17:
            from scripts.verify_th17_original_credit import inspect
            stock=inspect(f,r,rank)
            stage=stock['stage2_reached']
            spares,bombs,power=r['remaining_lives'],stock['terminal_bombs'],None
        else:raise ValueError(f'Unqualified training title {title}')
        if type(stage) is not bool:raise ValueError('Stage outcome unverified')
        credit=used[key]
        if credit['difficulty']!=rank:raise ValueError('Learner rank differs')
        rows[f'TH{title} '+('Hard' if rank==2 else 'Lunatic')].append(dict(seed=key[1],original_receipt_sha256=digest(f),
            decisions=credit['steps'],native_rank_samples=evidence['difficulty_verified_native_samples'],stage2=stage,
            terminal_spares=max(0,spares),terminal_bombs=bombs,terminal_power=power,
            **({'last_playable_bombs':stock['last_playable_bombs']} if title==17 else {})))
    if seen!=wanted:raise ValueError('Missing original credit')
    stats={}
    for k,group in sorted(rows.items()):
        end={f:mean(r[f] for r in group) if group[0][f] is not None else None for f in ('terminal_spares','terminal_bombs','terminal_power')}
        if 'last_playable_bombs' in group[0]:end['last_playable_bombs']=mean(r['last_playable_bombs'] for r in group)
        stats[k]=dict(complete_credits=len(group),decisions=sum(r['decisions'] for r in group),native_rank_samples=sum(r['native_rank_samples'] for r in group),
                      stage2=sum(r['stage2'] for r in group),mean_terminal_resources=end)
    report=dict(note='These are training trajectories of the frozen SFT behavior. Use the separately registered independent, matched, complete-credit test to compare SFT with the resulting PPO.',titles_and_ranks=stats,
                trained_model=learner['policy_sha256'],behavior=learner['native_behavior_sha256'],equal_weight_strata=learner['equal_weight_strata'],
                accepted_epochs=sum(x['accepted'] for x in learner['history']),unique_decisions=learner['steps'],credits=rows)
    a.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(stats),flush=True)


if __name__=='__main__':main()
