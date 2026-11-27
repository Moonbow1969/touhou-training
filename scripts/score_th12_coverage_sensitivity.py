"""Recount both four-title evaluation cohorts with original policy coverage adjudicated independently."""
import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from scripts.checkpoint_storage import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True,type=Path)
    parser.add_argument('--merged-as-deployed',type=Path,help='Optional complete merged-cohort results, including legacy coverage failures.')
    args=parser.parse_args()
    root=args.data.resolve()
    audit=root/'th12-control-coverage/strict-native-policy-coverage.json'
    verdicts=json.loads(audit.read_bytes())['credits']
    summaries={}
    files={b:root/b/'evaluation/credits.json' for b in ('four-game-online','four-game-ppo-generation2')}
    if args.merged_as_deployed:files['four-game-merged-online']=args.merged_as_deployed
    for branch,file in files.items():
        rows=[r for r in json.loads(file.read_bytes()) if r['title']==12]
        ver={v['sha256']:v for v in verdicts if v['branch']==branch and v['split']=='held'}
        if set(ver)!={r['native_log_sha256'] for r in rows}:
            raise ValueError('Missing exact policy-coverage evidence for released TH12 rows')
        plan=json.loads((root/branch/'camera-protocol.json').read_bytes())
        arms=(plan['reference_arm'],plan['candidate'])
        result={}
        for rank,label in ((2,'Hard'),(3,'Lunatic')):
            ranked=[r for r in rows if r['difficulty']==rank]
            groups={arm:{r['seed']:r for r in ranked if r['arm']==arm} for arm in arms}
            if groups[arms[0]].keys()!=groups[arms[1]].keys():
                raise ValueError('Unpaired original cohort')
            kept=[s for s in groups[arms[0]] if all(ver[groups[a][s]['native_log_sha256']]['coverage']=='qualified' for a in arms)]
            item=dict(physical_pairs=len(groups[arms[0]]),coverage_qualified_pairs=len(kept),qualified_paired_seeds=kept)
            for arm,by_seed in groups.items():
                def counts(selected):
                    if not selected:return dict(n=0,stage2=0,unknown_stage=0,mean_terminal_bombs=None,mean_terminal_spares=None)
                    return dict(n=len(selected),stage2=sum(x['stage2_reached'] is True for x in selected),
                                unknown_stage=sum(x['stage2_reached'] is None for x in selected),
                                mean_terminal_bombs=mean(x['terminal_bombs'] for x in selected),
                                mean_terminal_spares=mean(x['terminal_spares'] for x in selected))
                item[arm]=dict(coverage_counts=dict(Counter(ver[r['native_log_sha256']]['coverage'] for r in by_seed.values())),
                               as_deployed=counts(list(by_seed.values())),both_arms_qualified=counts([by_seed[s] for s in kept]))
            result[label]=item
        summaries[branch]=dict(original_held_sha256=digest(file),by_rank=result,
            original_verdicts={h:{k:v[k] for k in ('tag','coverage','signed_evidence_sha256','reason')} for h,v in ver.items()})
    out=dict(scope='Retrospective controller-coverage sensitivity on the originally frozen full-game evaluations. As-deployed includes all real original endpoints. The both-qualified subset retains a seed only if both policies controlled their complete playable credits. Inclusion can depend on achieved gameplay, so conditional outcomes describe that subset.',
             coverage_audit_sha256=digest(audit),cohorts=summaries)
    (audit.parent/'released-pair-sensitivity.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    for branch,s in summaries.items():
        print(branch,{k:v['coverage_qualified_pairs'] for k,v in s['by_rank'].items()},flush=True)


if __name__=='__main__':
    main()
