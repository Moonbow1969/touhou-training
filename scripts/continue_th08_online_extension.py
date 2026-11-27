"""Continue original TH08 physical lanes under the registered technical recovery and reserve rules."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from scripts.checkpoint_storage import digest
from scripts.qualify_th08_trial_credit import qualify
from scripts.recover_th08_native_terminal import recover
from scripts.train_five_original_games import freeze


def read(path):
    return json.loads(path.read_bytes())


def atomic_json(path,value):
    if path.exists():
        freeze(path,value)
    else:
        tmp=path.with_suffix('.pending')
        freeze(tmp,value)
        tmp.replace(path)


def selected_credit(root,folder,model,arm_sha,actual_seed,rank,role,out):
    destination=root/'certified-first-terminal'/folder.name
    physical=read(folder/'receipt.json')
    if physical.get('role')!=role or physical['policy_sha256']!=arm_sha or physical['seed']!=actual_seed:
        raise ValueError('Unexpected role or original behavior at technical selection')
    native,why=recover(folder,destination)
    if native is None:
        return None,dict(seed=actual_seed,source_sha256=digest(folder/'receipt.json'),physical_actions=physical['physical_decisions'],**why)
    row,audit=qualify(native,model,arm_sha,actual_seed,rank,role,out)
    return dict(record=row,audit=audit,qualified_source=native.relative_to(root).as_posix(),
                original_source=folder.relative_to(root).as_posix(),original_receipt_sha256=digest(folder/'receipt.json')),None


def launch(assets,policy,folder,rank,seed,role):
    subprocess.run([sys.executable,'-u','-m','scripts.run_th08_trial_camera','--assets',str(assets),
                    '--policy',str(policy),'--out',str(folder),'--rank',rank,'--seed',str(seed),
                    '--role',role,'--verified-menu'],check=True)


def launch_after_entry_failures(assets,policy,base,rank,seed,role):
    history=[]
    for number in range(1,4):
        folder=base if number==1 else base.with_name(base.name+f'-entry-{number}')
        if (folder/'receipt.json').is_file():
            return folder,history
        if (folder/'protocol.json').exists():
            if (folder/'first-policy-frame.png').exists():
                raise RuntimeError(f'Unfinished original gameplay in {folder.name}')
        else:
            try:
                if (base/'original-stream').exists() and folder!=base:
                    folder.mkdir(exist_ok=True)
                    import os
                    if not (folder/'original-stream').exists():os.link(base/'original-stream',folder/'original-stream')
                launch(assets,policy,folder,rank,seed,role)
                return folder,history
            except subprocess.CalledProcessError:
                if (folder/'first-policy-frame.png').exists():
                    raise
        history.append(dict(folder=folder.name,policy_decisions=0,
                            protocol_sha256=digest(folder/'protocol.json'),reason='Game worker exited before first policy observation'))
    raise RuntimeError(f'Three failed game entries for {seed}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('data','assets'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--only-rank',choices=('Hard','Lunatic'))
    a=p.parse_args();data=a.data.resolve();assets=a.assets.resolve();root=data/'six-game-online'
    plan=read(root/'protocol.json');amend=read(root/'technical-menu-amendment.json')
    equivalent=root/'runner-source-equivalence.json'
    accepted_sources={amend['revised_runner_sha256']}
    if equivalent.exists():
        record=read(equivalent)
        if record['registered_revised_sha256']!=amend['revised_runner_sha256']:
            raise ValueError('Unrelated runner equivalence')
        accepted_sources.add(record['canonical_revised_sha256'])
    if (digest(Path(__file__).with_name('run_th08_trial_camera.py')) not in accepted_sources
            or digest(root/'protocol.json')!=amend['original_registration_sha256']):
        raise ValueError('Expected the exact technical menu amendment')
    policy=data/'five-game-generation2/policy/policy.ts'
    if digest(policy)!=plan['parent_actor']:raise ValueError('Unexpected behavior')
    staging=root/'selected-training';staging.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    def lane(rank):
        model=torch.jit.load(str(policy),map_location='cpu').eval()
        original_stopped=False
        for slot in plan['th08_train_seeds']:
            if plan['rank_by_seed'][str(slot)]!=rank:continue
            output=staging/f'{slot}.json'
            original=root/'native08'/f'train-five-ppo2-{slot}'
            if output.exists():
                if (original/'receipt.json').is_file() and read(original/'receipt.json')['outcome']!='original_continue_prompt':
                    original_stopped=True
                continue
            rejected=[];entries=[]
            for attempt in range(3):
                seed=slot+attempt*10000
                folder=root/'native08'/f'train-five-ppo2-{seed}'
                receipt=folder/'receipt.json'
                if not receipt.exists():
                    if attempt==0 and not original_stopped:
                        while not receipt.is_file():
                            time.sleep(5)
                    else:
                        folder,entry_failures=launch_after_entry_failures(assets,policy,folder,rank,seed,'train')
                        entries.extend(entry_failures)
                        receipt=folder/'receipt.json'
                physical=read(receipt)
                if attempt==0 and physical['outcome']!='original_continue_prompt':
                    original_stopped=True
                result,bad=selected_credit(root,folder,model,plan['parent_actor'],seed,rank,'train',root/'th08-training')
                if bad:
                    rejected.append(bad)
                    atomic_json(staging/f'{slot}-rejected-{seed}.json',bad)
                    print('Excluded incomplete physical attempt',slot,seed,bad['reason'],flush=True)
                    continue
                result.update(slot=slot,selected_seed=seed,rejected_attempts=rejected,technical_entries=entries,
                              registration_sha256=digest(root/'technical-menu-amendment.json'))
                atomic_json(output,result)
                print('Selected physical credit',slot,seed,result['audit']['decisions'],flush=True)
                break
            else:
                raise ValueError(f'Both registered reserves exhausted for {slot}')
    with ThreadPoolExecutor(2) as pool:
        tasks=[pool.submit(lane,rank) for rank in ((a.only_rank,) if a.only_rank else ('Hard','Lunatic'))]
        for task in tasks:task.result()


if __name__=='__main__':
    main()
