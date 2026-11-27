"""Collect and verify the TH10 arm of the multi-title human replay PPO campaign."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time
from types import SimpleNamespace

import torch

from scripts.collect_th10_full_action import install_weights, one, pin, save, start_camera_service
from scripts.package_th10_credit_logs import package
from scripts.run_th10_camera_cohort import checkpoint, digest, verify
from scripts.run_th10_camera_pair import run as paired


def wait_for(path):
    while not path.is_file():
        time.sleep(4)


def host_wave(host, client, policy, seeds, directory, gate, lanes):
    wait_for(gate)
    sha = digest(policy)
    pinned = pin(client, camera=True)
    install_weights(client, policy, sha)
    start_camera_service(client, pinned, sha)
    progress = dict(host=host, seeds=seeds, policy_sha256=sha, completed=[])
    manifest = directory / f'progress-{host}.json'
    save(manifest, progress)
    with ThreadPoolExecutor(lanes) as pool:
        jobs = [pool.submit(one, (client, pinned, sha, host, seed, policy.parent.name,
                                  directory, 120, False)) for seed in seeds]
        for result in as_completed(jobs):
            credit = result.result()
            if not credit['terminal'] or credit['continues'] or credit['policy_sha256'] != sha:
                raise ValueError(f'Unexpected original endpoint on {host} seed {credit["seed"]}')
            progress['completed'].append(credit['seed'])
            save(manifest, progress)


def run(root, worker_file, lanes):
    root, worker_file = root.resolve(), worker_file.resolve()
    torch.set_num_threads(2)
    clients = json.loads(worker_file.read_bytes())
    gates = {'accelerator': root / 'th10-spend-ablation/native10/paired.json',
             'momentum': root / 'cross-title-online/native10/paired.json'}
    wait_for(root / 'campaign/g04/native10/ppo-receipts.json')
    for generation in (2, 3):
        group = root / f'cross-campaign/g{generation:02d}'
        protocol = json.loads((group / 'camera-protocol.json').read_bytes())
        seeds = protocol['train_seeds']['10']
        out = group / 'native10'
        out.mkdir(exist_ok=True)
        policy = root / protocol['registered_parent'] / 'policy.ts'
        wait_for(policy.parent / 'report.json')
        model = checkpoint(policy)
        assert digest(policy) == json.loads((policy.parent / 'report.json').read_bytes())['policy_sha256']
        save(out / 'training-protocol.json', dict(title=10, seeds=seeds, parent=protocol['registered_parent'],
             policy_sha256=digest(policy), host='Odd seeds Accelerator, even Momentum', lanes_per_host=lanes,
             cadence_ms=120, build='TH10 original official 0.02a trial, Normal, stock lives, no Continue',
             action='Sample all four heads from original pixels; native life, score and Power are outcome sensors.'))
        with ThreadPoolExecutor(2) as fleet:
            jobs = [fleet.submit(host_wave, host, clients[host], policy,
                                 [s for s in seeds if s % 2 == parity], out, gates[host], lanes)
                    for host, parity in [('accelerator', 1), ('momentum', 0)]]
            for job in as_completed(jobs):
                job.result()
        verify(out, policy, model, seeds, 'train', host='fleet')
        package(group / 'th10-native-logs.zip', [out])

    paired(SimpleNamespace(out=out, workers=worker_file, host='fleet',
        reference=root / protocol['reference_arm'] / 'policy.ts',
        candidate=root / protocol['candidate'] / 'policy.ts',
        reference_name='held-ppo1', candidate_name='held-ppo3',
        seeds=protocol['eval_seeds']['10'], concurrency=2*lanes, candidate_history=False))
    package(group / 'th10-held-native-logs.zip', [out / 'held-ppo1', out / 'held-ppo3'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--workers', required=True, type=Path)
    parser.add_argument('--lanes-per-host', default=2, type=int)
    args = parser.parse_args()
    run(args.root, args.workers, args.lanes_per_host)
