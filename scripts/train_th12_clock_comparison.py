"""Fit both registered TH12 cadence arms on equal decision budgets and score fresh native credits."""
import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from scripts.checkpoint_storage import digest


def freeze(path, value):
    if path.exists():
        if json.loads(path.read_bytes()) != value:
            raise ValueError(f'Registration changed at {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')
    temporary.replace(path)


def run(module, *args):
    subprocess.run([sys.executable, '-u', '-m', 'scripts.' + module, *map(str, args)], check=True)


def wait_for_receipts(root, rows, fingerprints):
    counts = Counter()
    for row in rows:
        path = root / row['cohort'] / 'native12/credits' / (row['tag'] + '.json')
        while not path.is_file():
            time.sleep(10)
        receipt = json.loads(path.read_bytes())
        key = row['policy']
        if (receipt['episode'] != row['tag'] or receipt['seed'] != row['seed']
                or receipt['difficulty'] != row['rank'] or receipt['policy_sha256'] != fingerprints[key]
                or not receipt.get('terminal', True) or receipt.get('continues', 0) != 0):
            raise ValueError(f'Original credit differs from registered cell {path}')
        counts[row['cohort'], row['rank']] += receipt['actions']
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    args = parser.parse_args()
    data = args.data.resolve()
    root = data / 'th12-clock-online'
    source = root / 'clock-protocol.json'
    plan = json.loads(source.read_bytes())
    parent = data / plan['parent']
    if digest(parent) != plan['parent_sha256']:
        raise ValueError('Deployment differs from the registered SFT')
    train = [r for r in plan['plan'] if r['mode'] == 'train']
    held = [r for r in plan['plan'] if r['mode'] == 'held']
    train_seeds = sorted({r['seed'] for r in train})
    held_seeds = sorted({r['seed'] for r in held})
    if set(train_seeds) & set(held_seeds):
        raise ValueError('Repeated training seed in evaluation')
    registration = dict(native_protocol_sha256=digest(source), train_seeds={'12': train_seeds},
                        eval_seeds={'12': held_seeds}, balanced_title_rank=True,
                        rank_by_seed={'12': {str(r['seed']): r['rank'] for r in plan['plan']}},
                        reference_arm='wall120-ppo', candidate='tick4-ppo',
                        learning_rates=[2e-5, 1e-5, 5e-6, 2.5e-6],
                        optimizer_budget='min(1536, available decisions in each of the four clock by rank cells). Full-credit GAE precedes sampling.')
    freeze(root / 'learner-registration.json', registration)
    counts = wait_for_receipts(root, train, {'trial-augmented': digest(parent)})
    arms = ('wall120', 'tick4')
    budget = min(1536, *(counts[arm, rank] for arm in arms for rank in (2, 3)))
    if budget < 2:
        raise ValueError(f'Missing clock by rank cell {counts}')
    protocol = root / 'optimizer-protocol.json'
    freeze(protocol, dict(registration, matched_decisions_per_stratum=budget))
    policies = {}
    for arm in arms:
        cohort = root / arm
        packed = cohort / 'camera-batch'
        if not (packed / 'credits.json').is_file():
            run('pack_camera_ppo_rollouts', '--root', cohort, '--round2', '--behavior', parent,
                '--protocol', protocol, '--out', packed)
        output = cohort / 'policy'
        if not (output / 'report.json').is_file():
            run('train_full_resolution_camera_ppo', '--manifest', packed / 'credits.json',
                '--protocol', protocol, '--behavior', parent, '--sft', parent.with_name('camera-policy.pt'), '--out', output)
        report = json.loads((output / 'report.json').read_bytes())
        if (report['protocol_sha256'] != digest(protocol) or report['native_behavior_sha256'] != digest(parent)
                or report['policy_sha256'] != digest(output / 'policy.ts')
                or any(s['selected'] != budget for s in report['optimizer_decision_sampling'].values())):
            raise ValueError('Optimizer record differs from registered budget or actor')
        policies[arm + '-ppo'] = report['policy_sha256']
        print(arm, report['steps'], 'native decisions;', 2 * budget, 'optimizer rows', flush=True)
    wait_for_receipts(root, held, policies)
    out = root / 'evaluation'
    run('score_original_camera_run', '--root', root / 'common-tick4', '--round2', '--protocol', protocol,
        '--parent', root / 'wall120/policy/policy.ts', '--candidate', root / 'tick4/policy/policy.ts',
        '--arm-names', 'wall120-ppo', 'tick4-ppo', '--out', out)
    run('plot_full_action_results', '--folder', out, '--candidates', 'tick4-ppo')


if __name__ == '__main__':
    main()
