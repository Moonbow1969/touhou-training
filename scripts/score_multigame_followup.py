"""Score matched five-title and six-title follow-ups after their physical test lanes return."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.checkpoint_storage import digest
from scripts.score_original_camera_run import records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--study',choices=('five-game-crossfit-advantages','six-game-retention','six-game-return-horizon','six-game-bomb-cost','full-epochs-original','reviewed-five-human-online','feedback-ppo-independent-replication','chained-five-game-ppo'),required=True)
    a=p.parse_args();d=a.data.resolve();root=d/a.study
    while not (root/'camera-protocol.json').exists():time.sleep(10)
    plan=json.loads((root/'camera-protocol.json').read_bytes())
    five=a.study=='five-game-crossfit-advantages'
    horizon=a.study=='six-game-return-horizon'
    reference=(root/plan['reference_arm'] if horizon else d/'five-game-generation3')/'policy/policy.ts'
    candidate=(root/plan['candidate'] if horizon else root if five else d/'six-game-online/joint-six')/'policy/policy.ts'
    if a.study=='six-game-bomb-cost':
        reference=d/'six-game-online/joint-six/policy/policy.ts'
        candidate=root/'policy/policy.ts'
    if a.study=='reviewed-five-human-online':
        reference=d/'first-five-title-retail-sft/five-reviewed/policy.ts'
        candidate=root/'policy/policy.ts'
    if a.study=='full-epochs-original':
        reference=d/'four-title-complete-epochs'/plan['reference_arm']/'policy.ts'
        candidate=d/'four-title-complete-epochs'/plan['candidate']/'policy.ts'
    if a.study=='feedback-ppo-independent-replication':
        reference=d/'four-title-four-step-feedback/teacher/policy.ts'
        candidate=d/'four-step-feedback-online/policy/policy.ts'
        if digest(reference)!=plan['behavior_sha256'] or digest(candidate)!=plan['candidate_sha256']:
            raise ValueError('Replication actors changed')
    if a.study=='chained-five-game-ppo':
        reference=d/'reviewed-five-human-online/policy/policy.ts'
        candidate=root/'policy/policy.ts'
    while not candidate.with_name('report.json').is_file():time.sleep(15)
    refs={digest(reference),digest(candidate)}
    titles=sorted(int(t) for t in plan['eval_seeds'] if t!='8')
    expected={(t,s,h) for t in titles for s in plan['eval_seeds'][str(t)] for h in refs}
    previous=-1
    while True:
        seen={}
        for t,file in records(root,True,True,include_held_endpoints=True):
            try:r=json.loads(file.read_bytes())
            except (json.JSONDecodeError,PermissionError):continue
            key=(t,r.get('seed'),r.get('policy_sha256'))
            if key not in expected or (t==13 and not r.get('ranking_photo')):continue
            if key in seen:raise ValueError('Duplicate complete physical receipt')
            seen[key]=file
        if len(seen)!=previous:
            print('Original held games',len(seen),'/',len(expected),flush=True)
            previous=len(seen)
        if seen.keys()==expected:break
        time.sleep(20)
    if 12 in titles:
        coverage=root/'th12-held-qualified-coverage.json'
        physical=[f for (t,s,h),f in seen.items() if t==12]
        while not coverage.exists():time.sleep(10)
        rows=json.loads(coverage.read_bytes())['credits']
        expected_status={digest(f):'indeterminate' if f.parent.name=='endpoint-only' else 'qualified' for f in physical}
        if len(rows)!=len(physical) or {z['receipt_sha256']:z['coverage'] for z in rows}!=expected_status or any(not z['evidence_sha256'] for z in rows):
            raise ValueError('Original TH12 input coverage and endpoint populations do not reconcile')
    def run(module,*args):
        subprocess.run([sys.executable,'-m','scripts.'+module,*map(str,args)],check=True)
    out=root/'evaluation'
    run('score_original_camera_run','--root',root,'--round2','--protocol',root/'camera-protocol.json',
        '--parent',reference,'--candidate',candidate,'--arm-names',plan['reference_arm'],plan['candidate'],
        '--titles',*titles,'--out',out,'--include-held-endpoints')
    if 13 in titles:
        run('score_native_stage_clock','--credits',out/'credits.json','--title',13,'--arms',plan['reference_arm'],plan['candidate'],'--out',out/'native13-stage-clock.json')
    run('plot_full_action_results','--folder',out,'--candidates',plan['candidate'])
    for rank in ('Hard','Lunatic'):
        run('render_cross_title_credits','--root',root,'--out',out/f'first-{rank.lower()}-native-credits.mp4','--difficulty',rank,'--titles',*titles)
    if '8' in plan['eval_seeds']:
        held=root/'th08/evaluation/held-results.json'
        while not held.is_file():time.sleep(20)
        print('TH08 completed',digest(held),flush=True)


if __name__=='__main__':
    main()
