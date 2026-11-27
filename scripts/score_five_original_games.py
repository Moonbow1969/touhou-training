"""Reuse frozen reference credits and score each registered five-game comparison as it finishes."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from scripts.checkpoint_storage import digest
from scripts.run_registered_camera_update import wait_th12_coverage
from scripts.score_original_camera_run import records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--public',type=Path,required=True)
    args=parser.parse_args()
    data=args.data.resolve()
    root=data/'five-game-online'
    public=args.public.resolve()/'five-game-online'
    plan_file=root/'camera-protocol.json'
    plan=json.loads(plan_file.read_bytes())
    parent=data/'specialist-vs-shared/merged-equal/policy.ts'
    candidate=root/'policy/policy.ts'
    report=json.loads((root/'policy/report.json').read_bytes())
    hashes={digest(parent):plan['reference_arm'],digest(candidate):plan['candidate']}
    if digest(candidate)!=report['policy_sha256'] or digest(parent)!=plan['behavior_sha256']:
        raise ValueError('Policy identity changed')
    th11=data/'th11-original-ppo'
    qualification=json.loads((th11/'stage2-qualification.json').read_bytes())
    if not qualification['qualified'] or digest(th11/qualification['evidence_zip'])!=qualification['evidence_zip_sha256']:
        raise ValueError('TH11 stage qualification evidence changed')
    wanted={int(t):{(s,a) for s in seeds for a in hashes.values()} for t,seeds in plan['eval_seeds'].items()}
    def score(out, titles=()):
        subprocess.run([sys.executable,'-m','scripts.score_original_camera_run','--root',root,'--round2',
                        '--protocol',plan_file,'--parent',parent,'--candidate',candidate,'--out',out,
                        '--arm-names',plan['reference_arm'],plan['candidate'],
                        *(['--titles',*map(str,titles)] if titles else [])],check=True)
    def publish(folder):
        shutil.copytree(folder,public/folder.relative_to(root),dirs_exist_ok=True)
    done=set()
    last=None
    while done!=set(wanted):
        for old in (th11,data/'four-game-merged-online'):
            for t,receipt in records(old,True,True):
                if t==12 or t not in wanted:continue
                dest=root/receipt.relative_to(old)
                if dest.exists():continue
                row=json.loads(receipt.read_bytes())
                if ((row.get('seed'),plan['reference_arm']) not in wanted[t]
                        or row.get('policy_sha256')!=plan['behavior_sha256']):continue
                if t==13 and not row.get('ranking_photo'):continue
                dest.parent.mkdir(parents=True,exist_ok=True)
                for f in receipt.parent.iterdir():
                    if f.is_file() and f!=receipt and not (dest.parent/f.name).exists():
                        os.link(f,dest.parent/f.name)
                os.link(receipt,dest)
        found={t:{} for t in wanted}
        for t,path in records(root,True,True):
            row=json.loads(path.read_bytes())
            key=(row.get('seed'),hashes.get(row.get('policy_sha256')))
            if key not in wanted[t] or (t==13 and not row.get('ranking_photo')):continue
            if key in found[t]:raise ValueError(f'Duplicate native credit {t} {key}')
            found[t][key]=path
        count={t:len(v) for t,v in found.items()}
        if count!=last:
            print('Verified policy/seed receipts by title',count,flush=True)
            last=count
        for t in sorted(wanted.keys()-done):
            if set(found[t])!=wanted[t]:continue
            if t==12:
                coverage=root/'th12-held-qualified-coverage.json'
                if not coverage.exists():continue
                wait_th12_coverage(coverage,found[t].values())
            out=root/'evaluation'/f'th{t}'
            score(out,[t])
            publish(out)
            done.add(t)
        if done!=set(wanted):time.sleep(20)
    evaluation=root/'evaluation'
    score(evaluation)
    subprocess.run([sys.executable,'-m','scripts.summarize_four_title_pair','--root',root],check=True)
    publish(evaluation)
    for rank in ('Hard','Lunatic'):
        film=root/'recordings'/f'five-games-{rank.lower()}.mp4'
        subprocess.run([sys.executable,'-m','scripts.render_cross_title_credits','--root',root,
                        '--out',film,'--difficulty',rank],check=True)
    publish(root/'recordings')


if __name__=='__main__':
    main()
