"""Preserve physical outcomes, conditional TH12 evidence and unaffected game comparisons."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.checkpoint_storage import digest
from scripts.score_original_camera_run import records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--public',type=Path,required=True)
    args=parser.parse_args()
    data=args.data.resolve()
    root=data/'four-game-merged-online'
    plan=json.loads((root/'camera-protocol.json').read_bytes())
    parent=data/'specialist-vs-shared/merged-equal/policy.ts'
    candidate=root/'policy/policy.ts'
    hashes={digest(parent),digest(candidate)}
    eligible={int(t):set(seeds) for t,seeds in plan['eval_seeds'].items()}
    done=set()
    while done!=set(eligible):
        found={t:set() for t in eligible}
        for t,f in records(root,True,True):
            row=json.loads(f.read_bytes())
            if row['seed'] in eligible[t] and row.get('policy_sha256') in hashes and (t!=13 or row.get('ranking_photo')):
                found[t].add((row['seed'],row['policy_sha256']))
        for t in sorted(eligible.keys()-done):
            if len(found[t])!=2*len(eligible[t]):continue
            name='as-deployed-th12' if t==12 else f'qualified-th{t}'
            folder=root/name
            subprocess.run([sys.executable,'-m','scripts.score_original_camera_run','--root',root,'--round2','--protocol',
                root/'camera-protocol.json','--parent',parent,'--candidate',candidate,'--arm-names',plan['reference_arm'],
                plan['candidate'],'--titles',str(t),'--out',folder],check=True)
            if t==12:
                (folder/'README.md').write_text('# Actual TH12 deployments under the legacy controller\n\nThese physical full-credit results include episodes with missing policy actions during a playable boss spell. The coverage-qualified comparison requires both arms to pass the independent per-credit review. See the [coverage audit](../../th12-control-coverage/released-pair-sensitivity.json).\n',encoding='utf8')
            shutil.copytree(folder,args.public/root.name/name,dirs_exist_ok=True)
            done.add(t)
            print('Sealed original endpoints',t,name,flush=True)
        if done!=set(eligible):time.sleep(25)
    files=json.loads((root/'as-deployed-th12/credits.json').read_bytes())
    expected={r['native_log_sha256'] for r in files}
    directory=data/'th12-control-coverage'
    audit=directory/'strict-native-policy-coverage.json'
    while True:
        verdicts=[v for v in json.loads(audit.read_bytes())['credits'] if v['branch']==root.name and v['split']=='held']
        if {v['sha256'] for v in verdicts}==expected:break
        time.sleep(25)
    subprocess.run([sys.executable,'-m','scripts.score_th12_coverage_sensitivity','--data',data,
                    '--merged-as-deployed',root/'as-deployed-th12/credits.json'],check=True)
    out=args.public/directory.name
    for name in ('released-pair-sensitivity.json','strict-native-policy-coverage.json','additional-reviewed-positive.json'):
        shutil.copy2(directory/name,out/name)


if __name__=='__main__':
    main()
