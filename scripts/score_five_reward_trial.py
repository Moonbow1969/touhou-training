"""Score fresh TH11 reward-ablation credits with a verified original stage and stock stream."""
import argparse
import json
import shutil
from pathlib import Path

from scripts.checkpoint_storage import digest
from scripts.train_th11_original_ppo import score_held,wait_file


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--public',type=Path,required=True)
    args=parser.parse_args();data=args.data.resolve()
    root=data/'five-game-th11-score';source=data/'th11-original-ppo'
    plan=json.loads((root/'camera-protocol.json').read_bytes())
    check=digest(data/'five-game-online/policy/policy.ts')
    if check!=plan['reference_evaluation_sha256']:raise ValueError('Reference actor changed')
    wait_file(root/'policy/report.json')
    report=json.loads((root/'policy/report.json').read_bytes())
    candidate=digest(root/'policy/policy.ts')
    if candidate!=report['policy_sha256']:raise ValueError('Candidate actor changed')
    q=json.loads((source/'stage2-qualification.json').read_bytes())
    for name in ('stage2-qualification.json',q['evidence_zip']):
        local=root/name
        if not local.exists():shutil.copy2(source/name,local)
        if digest(local)!=digest(source/name):raise ValueError('Different stage evidence')
    score_held(root,plan,check,candidate)
    shutil.copytree(root/'evaluation',args.public/root.name/'evaluation',dirs_exist_ok=True)


if __name__=='__main__':
    main()
