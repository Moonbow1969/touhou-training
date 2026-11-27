"""Count genuine per-title and per-rank transitions for the four-game policy series and TH11."""
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
    studies=['four-game-online','four-game-ppo-generation2','four-game-merged-online','th11-original-ppo']
    counts=Counter();seen=set();reports={}
    for study in studies:
        path=data/study/'policy/report.json'
        if not path.exists():continue
        result=json.loads(path.read_bytes())
        if result['policy_sha256']!=digest(path.with_name('policy.ts')):
            raise ValueError(f'Unverified fitted {study} checkpoint')
        reports[study]=dict(learner_report_sha256=digest(path),checkpoint=result['policy_sha256'],steps=result['steps'])
        for c in result['credits']:
            key=(c['title'],c['seed'],result['native_behavior_sha256'])
            if key in seen:raise ValueError(f'Duplicate original credit {key}')
            seen.add(key)
            counts[c['title'],c['difficulty'],'credits']+=1
            counts[c['title'],c['difficulty'],'decisions']+=c['steps']
    table={f'TH{t}':{name:{k:counts[t,r,k] for k in ('credits','decisions')} for r,name in ((2,'Hard'),(3,'Lunatic'))}
           for t in sorted({k[0] for k in counts})}
    out=dict(count_unit='Full ordinary original training credit with one specific frozen behavior policy. Counts are from fitted PPO checkpoints. Four-game first update, its continuation, the merged-human-SFT four-game branch and the independent TH11 trial branch. The three-game subset sibling uses the same native behavior and contributes no additional credits.',
             credits=len(seen),decisions=sum(v for (t,r,key),v in counts.items() if key=='decisions'),
             by_game=table,studies=reports)
    path=data/'learning-curves/hard-lunatic-training-coverage.json'
    path.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    print(out['credits'],out['decisions'],flush=True)


if __name__=='__main__':
    main()
