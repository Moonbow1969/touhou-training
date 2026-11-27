"""Advance one frozen stock-game rollout batch through PPO and matched evaluation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.checkpoint_storage import digest
from scripts.score_original_camera_run import records, verified_th10_entry


def wait_th12_coverage(path, files):
    """Bind a stage/terminal input-coverage adjudication to each exact original receipt."""
    while not path.is_file():
        time.sleep(15)
    report=json.loads(path.read_bytes())
    expected={digest(f) for f in files}
    rows=report['credits']
    if (not expected or len(rows)!=len(expected)
            or {r['receipt_sha256'] for r in rows}!=expected
            or any(r['coverage']!='qualified' or not r.get('evidence_sha256') for r in rows)):
        raise ValueError(f'Unresolved playable control coverage in {path}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','parent','protocol'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--from-sft',action='store_true',help='Restore the human camera policy with a zero-initialized value head.')
    p.add_argument('--th12-coverage',action='store_true',help='Require per-credit playable-control coverage before fitting and publishing held outcomes.')
    args=p.parse_args()
    root=args.root.resolve()
    plan=json.loads(args.protocol.read_bytes())
    behavior=args.parent/'policy.ts'
    fingerprint=digest(behavior)
    if fingerprint!=plan['behavior_sha256']:
        raise ValueError('Unexpected parent actor')
    if args.from_sft:
        supervised=json.loads((args.parent/'report.json').read_bytes())
        if supervised['sft']!=digest(args.parent/'camera-policy.pt') or supervised['torchscript_sha256']!=fingerprint:
            raise ValueError('Supervised training and deployed behavior disagree')
    reference=plan['reference_arm']
    candidate=plan['candidate']
    def run(module,*params):
        subprocess.run([sys.executable,'-u','-m','scripts.'+module,*map(str,params)],check=True)
    wanted={(int(t),s) for t,seeds in plan['train_seeds'].items() for s in seeds}
    seen={}
    while set(seen)!=wanted:
        start=len(seen)
        for title,f in records(root,True,True):
            try:
                r=json.loads(f.read_bytes())
            except (json.JSONDecodeError,PermissionError):
                continue
            key=title,r.get('seed')
            if key not in wanted:continue
            if key in seen:
                if seen[key]!=f:raise ValueError('Duplicate training seed')
                continue
            if r['policy_sha256']!=fingerprint or not r.get('terminal',True) or r.get('continues',0):
                raise ValueError('Training receipt violates registered behavior or original stock')
            if title==10 and verified_th10_entry(r,f.parent)[0] is None:
                raise ValueError('Training credit used the retired TH10 stage channel')
            seen[key]=f
        if len(seen)!=start:print('Verified full training credits',len(seen),'/',len(wanted),flush=True)
        if set(seen)!=wanted:time.sleep(15)
    if args.th12_coverage:
        wait_th12_coverage(root/'th12-train-qualified-coverage.json',[f for (t,_),f in seen.items() if t==12])
    if not (root/'camera-batch/credits.json').is_file():
        run('pack_camera_ppo_rollouts','--root',root,'--round2','--protocol',args.protocol,'--behavior',behavior,'--out',root/'camera-batch')
    if not (root/'corrected/clock-audit.json').is_file():
        run('normalize_th13_survival_clock','--original',root,'--out',root/'corrected')
    policy=root/'policy'
    if not (policy/'report.json').is_file():
        restore=('--sft',args.parent/'camera-policy.pt') if args.from_sft else ('--parent-actor-critic',args.parent/'camera-ppo.pt')
        run('train_full_resolution_camera_ppo','--manifest',root/'corrected/credits.json','--protocol',args.protocol,
            *restore,'--behavior',behavior,'--out',policy)
    result=json.loads((policy/'report.json').read_bytes())
    if result['native_behavior_sha256']!=fingerprint or result['policy_sha256']!=digest(policy/'policy.ts') or result['protocol_sha256']!=digest(args.protocol):
        raise ValueError('Trained checkpoint or registration differs')
    out=root/'evaluation'
    if args.th12_coverage:
        seeds=set(plan['eval_seeds']['12'])
        hashes={fingerprint,result['policy_sha256']}
        files={}
        while len(files)!=len(seeds)*len(hashes):
            for title,f in records(root,True,True):
                if title!=12:continue
                receipt=json.loads(f.read_bytes())
                key=receipt['seed'],receipt['policy_sha256']
                if key[0] in seeds and key[1] in hashes:
                    if key in files and files[key]!=f:raise ValueError('Duplicate TH12 evaluation receipt')
                    files[key]=f
            if len(files)!=len(seeds)*len(hashes):time.sleep(15)
        wait_th12_coverage(root/'th12-held-qualified-coverage.json',files.values())
    while True:
        proc=subprocess.run([sys.executable,'-m','scripts.score_original_camera_run','--root',str(root),'--round2','--protocol',str(args.protocol),
            '--parent',str(behavior),'--candidate',str(policy/'policy.ts'),'--arm-names',reference,candidate,'--out',str(out)],capture_output=True,text=True)
        if proc.returncode==0:break
        if 'Waiting for verified whole credits' not in proc.stderr:raise RuntimeError(proc.stderr)
        time.sleep(25)
    run('score_native_stage_clock','--credits',out/'credits.json','--title',13,'--arms',reference,candidate,'--out',out/'native13-stage-clock.json')
    run('plot_full_action_results','--folder',out,'--candidates',candidate)


if __name__=='__main__':
    main()
