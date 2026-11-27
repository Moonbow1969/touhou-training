"""Measure complete original TH10 credits for three frozen human replay SFT policies."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np
import torch

from scripts.collect_th10_full_action import install_weights, one, pin, save, start_camera_service
from scripts.package_th10_credit_logs import package
from scripts.run_th10_camera_cohort import checkpoint, digest, verify
from scripts.run_th10_camera_pair import switches
from scripts.summarize_th10_full_credit import draw, read_arm, summarize


def run(root, workers):
    torch.set_num_threads(2)
    root, workers = root.resolve(), workers.resolve()
    out = root / 'v2-supervised/native10'
    out.mkdir(parents=True, exist_ok=True)
    codes = {'A': 'cross-SFT', 'B': 'v2-3000', 'C': 'v2-9000'}
    policy = dict(zip(codes.values(), [root / 'camera-cross-sft/policy.ts',
        root / 'v2-supervised/joint-v2-3000/policy.ts', root / 'v2-supervised/joint-v2-9000/policy.ts']))
    models = {n: checkpoint(p) for n, p in policy.items()}
    hashes = {n: digest(p) for n,p in policy.items()}
    for name,p in policy.items():
        assert hashes[name] == json.loads((p.parent / 'report.json').read_bytes())['torchscript_sha256']
        (out / name).mkdir(exist_ok=True)
    seeds = list(range(895101,895109))
    orders = dict(zip(seeds, ('ABC','CBA','BCA','BAC','CAB','ACB','ACB','CBA')))
    hosts = {s: 'accelerator' if s % 2 else 'momentum' for s in seeds}
    description = ('Eight fresh controller seeds, one physical host per matched triple. All six arm orders are '
                   'represented; order by seed is preregistered. TH10 original 0.02a Normal trial, stock lives, '
                   'no Continue. Unassisted movement, focus, shooting and Spirit Strike sampled at temperature 1, '
                   '120 ms hold. Original game RNG is uncontrolled.')
    save(out / 'protocol.json', dict(description=description, host_by_seed=hosts, policy_sha256=hashes,
         order_by_seed={s:[codes[c] for c in orders[s]] for s in seeds}, train_samples='Human replay SFT only'))
    clients=json.loads(workers.read_bytes())

    def run_host(host):
        client=clients[host]
        pinned=pin(client,camera=True)
        for name, model_path in policy.items():
            install_weights(client,model_path,hashes[name])
            start_camera_service(client,pinned,hashes[name])
        def triple(seed):
            for code in orders[seed]:
                name=codes[code]
                row=one((client,pinned,hashes[name],host,seed,name,out/name,120,False))
                assert row['seed']==seed and row['terminal'] and row['continues']==0
            return seed
        with ThreadPoolExecutor(2) as lanes:
            futures=[lanes.submit(triple,s) for s in seeds if hosts[s]==host]
            completed=[]
            for job in as_completed(futures):
                completed.append(job.result())
                save(out / f'completed-{host}.json', completed)
    with ThreadPoolExecutor(2) as fleet:
        list(fleet.map(run_host, ['accelerator','momentum']))
    rows, movement = {},{}
    for name, model_path in policy.items():
        verify(out/name,model_path,models[name],seeds,'held',host='fleet')
        rows[name]=read_arm([out/name],seeds)
        movement[name]=switches(out/name,'fleet',seeds)
    result=summarize(rows,description,'cross-SFT')
    for name, arm in rows.items():
        first_loss=[arm[s]['loss_events'][0]['seconds'] for s in seeds]
        result['group'][name].update(mean_first_death_s=float(np.mean(first_loss)),
            median_first_death_s=float(np.median(first_loss)),
            movement_switch_rate=movement[name]['pooled_rate'],
            movement_hold_rate=1-movement[name]['pooled_rate'])
    result['movement_switches']=movement
    save(out/'paired.json',result)
    draw(rows,out/'paired.png')
    package(out/'native-logs.zip',[out/name for name in policy])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--workers',type=Path,required=True)
    args=parser.parse_args()
    run(args.root,args.workers)
