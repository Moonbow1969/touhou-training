"""Count distinct Hard and Lunatic original-game interactions by title before optimizer repetition."""
import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.checkpoint_storage import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True,type=Path)
    args=parser.parse_args()
    data=args.data.resolve()
    studies=['high-rank-online','four-game-online','four-game-ppo-generation2','four-game-merged-online',
             'th11-original-ppo','human78-online-covered','five-game-generation2','direction-feedback-original-ppo','five-game-generation3',
             'six-game-online/first-eight','six-game-online/joint-six','th08-multiteam-online/joint-six-four-teams',
             'later-human-online','six-game-return-horizon/six-gae985','four-step-feedback-online']
    counts=Counter();seen=set();steps_by_source={};reports={}
    for study in studies:
        path=data/study/'policy/report.json'
        if not path.exists():continue
        result=json.loads(path.read_bytes())
        if result['policy_sha256']!=digest(path.with_name('policy.ts')):
            raise ValueError(f'Unverified fitted {study} checkpoint')
        reports[study]=dict(learner_report_sha256=digest(path),checkpoint=result['policy_sha256'],steps=result['steps'])
        local=set()
        reused_sources=0
        for c in result['credits']:
            key=(c['title'],c['seed'],result['native_behavior_sha256'])
            if key in local:raise ValueError(f'Duplicate credit within one update {key}')
            local.add(key)
            if key in seen:
                if study not in ('six-game-online/joint-six','th08-multiteam-online/joint-six-four-teams') or steps_by_source[key]!=c['steps']:
                    raise ValueError(f'Unexpected repeated original credit {key}')
                reused_sources+=1
                continue
            seen.add(key);steps_by_source[key]=c['steps']
            counts[c['title'],c['difficulty'],'credits']+=1
            counts[c['title'],c['difficulty'],'decisions']+=c['steps']
        reports[study]['new_original_credits']=len(local)-reused_sources
        reports[study]['previously_counted_original_credits']=reused_sources
    table={f'TH{t:02}':{name:{k:counts[t,r,k] for k in ('credits','decisions')} for r,name in ((2,'Hard'),(3,'Lunatic'))}
           for t in sorted({k[0] for k in counts})}
    reused={}
    for study in ('merged-three-control-coverage','five-game-online','five-game-th11-score','five-game-crossfit-advantages',
                  'six-game-return-horizon/six-return','six-game-bomb-cost'):
        path=data/study/'policy/report.json'
        if not path.exists():continue
        result=json.loads(path.read_bytes())
        keys={(r['title'],r['seed'],result['native_behavior_sha256']) for r in result['credits']}
        if not keys<=seen or len(keys)!=len(result['credits']) or digest(path.with_name('policy.ts'))!=result['policy_sha256']:
            raise ValueError(f'Invalid source reuse in {study}')
        reused[study]=dict(reused_credits=len(keys),reused_decisions=result['steps'],checkpoint=result['policy_sha256'])
    out=dict(count_unit='Distinct original full-credit trajectories from main-series PPO reports carrying native Hard or Lunatic labels, identified by title, seed and frozen behavior. Decisions count optimizer input rows before epoch repetition. Historic TH10 inputs can include a small number of terminal overshoot rows. The six-title return-horizon study removed these rows before fitting. The original high-rank-online cohort predates the strict TH12 live-control audit. Additional fits on the same physical credits appear separately. New generations enter the ledger after fitting.',
             credits=len(seen),decisions=sum(v for (t,r,key),v in counts.items() if key=='decisions'),
             by_game=table,studies=reports,shared_trajectory_fits=reused)
    path=data/'learning-curves/hard-lunatic-training-coverage.json'
    path.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    print(out['credits'],out['decisions'],flush=True)


if __name__=='__main__':
    main()
