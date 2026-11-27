"""Resolve preassigned technical reserve slots and continue the original five-title PPO study."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    d=parser.parse_args().data.resolve();root=d/'five-game-generation2'
    original=root/'camera-protocol.json';rule_file=root/'th12-train-technical-amendment.json'
    amendment=json.loads(rule_file.read_bytes());plan=json.loads(original.read_bytes())
    if digest(original)!=amendment['original_protocol_sha256'] or digest(root/'th12-terminal-consumption-registration.json')!=amendment['terminal_consumption_registration_sha256']:
        raise ValueError('Amendment does not match registered original protocol')
    allocation={j['seed']:j for j in json.loads((root/'native12/protocol.json').read_bytes())['plan'] if j['mode']=='train'}
    expected=plan['train_seeds']['12']
    if set(expected)!=set(allocation) or expected!=amendment['original_seeds']:
        raise ValueError('Changed original seed slots')
    selection=root/'th12-train-source-selection.json'
    while not selection.exists():time.sleep(10)
    evidence=json.loads(selection.read_bytes());rows=evidence['rows']
    if len(rows)!=len(expected) or {r['original_seed'] for r in rows}!=set(expected):
        raise ValueError('Incomplete original TH12 slots')
    accepted={}
    rejected_count=0
    for r in rows:
        seed=r['original_seed'];source=r['selected_seed'];old=allocation[seed]
        if (old['host'],old['lane'],old['difficulty'])!=(r['host'],r['lane'],r['rank']):raise ValueError('Unmatched reserve lane or difficulty')
        if source is None:raise ValueError(f'Reserve attempts exhausted for {seed}')
        attempts=r['rejected_attempts'];stride=amendment['reserve_stride']
        if len(attempts)>amendment['max_reserve_attempts'] or source!=seed+len(attempts)*stride:
            raise ValueError('Unregistered reserve seed or attempt count')
        for i,failed in enumerate(attempts):
            f=(root/failed['evidence_path']).resolve()
            if (failed['seed']!=seed+i*stride or not failed['reason'] or not f.is_relative_to(root)
                    or digest(f)!=failed['evidence_sha256']):
                raise ValueError('Missing immutable evidence for skipped training slot')
        rejected_count+=len(attempts)
        file=root/'native12/credits'/f"train-{plan['reference_arm']}-{source}-{r['lane']}.json"
        credit=json.loads(file.read_bytes())
        if (digest(file)!=r['receipt_sha256'] or credit['policy_sha256']!=plan['behavior_sha256']
                or credit['difficulty']!=r['rank'] or credit['seed']!=source or credit['lane']!=r['lane']
                or credit['host'].replace('steamdeck','deck')!=r['host'] or credit.get('mode')!='train'):
            raise ValueError('Selected credit differs from original hardware, source or policy')
        accepted[seed]=source
    plan['train_seeds']['12']=[accepted[s] for s in expected]
    for s in expected:
        if s!=accepted[s]:
            plan['rank_by_seed']['12'][str(accepted[s])]=plan['rank_by_seed']['12'].pop(str(s))
    plan['technical_selection']=dict(original_registration_sha256=digest(original),amendment_sha256=digest(rule_file),
            selection_sha256=digest(selection),qualified_th12_credits=len(accepted),
            attempted_th12_credits=len(accepted)+rejected_count,rejected_attempts=rejected_count)
    effective=root/'effective-train-protocol.json';freeze(effective,plan)
    print('Selected original whole credits',plan['technical_selection'],flush=True)
    subprocess.run([sys.executable,'-u','-m','scripts.run_registered_camera_update','--root',root,
                    '--parent',d/'five-game-online/policy','--protocol',effective,'--th12-coverage'],check=True)


if __name__=='__main__':
    main()
