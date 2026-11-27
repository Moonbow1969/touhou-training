"""Train on eligible TH13 Easy final-spell suffixes and compare ordinary stock starts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.audit_th13_policy_suffix import verify
from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze


def checked_file(root, record, name):
    path = (root / record[name + '_path']).resolve()
    path.relative_to(root)
    if not path.is_file() or digest(path) != record[name + '_sha256']:
        raise ValueError(f'Changed {name} for seed {record["seed"]}')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    data = parser.parse_args().data.resolve()
    root = data / 'th13-visual-boss-curriculum'
    root.mkdir(exist_ok=True)
    engineering = data / 'engineering/th13-easy-visual-boss-handoff/protocol.json'
    stage = json.loads(engineering.read_bytes())
    model = data / 'specialist-vs-shared/merged-equal'
    actor = model / 'policy.ts'
    parent = json.loads((model / 'report.json').read_bytes())
    if digest(actor) != stage['policy_sha256'] or parent['sft'] != digest(model / 'camera-policy.pt'):
        raise ValueError('The training parent differs from the qualified handoff policy')
    attempts = list(range(1591301, 1591317))
    held = list(range(1591351, 1591367))
    plan = dict(reference_arm='merged-sft', candidate='boss-curriculum-ppo',
                behavior_sha256=digest(actor), camera_state_sha256=parent['sft'],
                engineering_protocol_sha256=digest(engineering), native_game_sha256=stage['game_sha256'],
                attempted_training_seeds={'13': attempts}, eval_seeds={'13': held},
                rank_by_seed={'13': {str(s): 0 for s in attempts + held}},
                balanced_title_rank=False, learning_rates=[2e-5, 1e-5, 5e-6, 2.5e-6],
                acquisition='Sixteen fresh stock Easy starts. Guide to a live final spell using the frozen photographic '
                            'gate. Release all guide keys. Include every adjudicated live-boss suffix regardless of '
                            'subsequent death or Stage 2. Preserve failed approaches in the overall denominator.',
                learning='All four SFT action heads and the vision encoder train with PPO for up to four accepted epochs. '
                         'The critic initializes at zero. Previous camera difference resets at handoff. No guide actions '
                         'receive PPO credit. Keep the native stock/cast/stage rewards, replacing the wall-time survival '
                         'term with the per-native-tick correction used by the whole-credit learner.',
                evaluation='Sixteen fresh paired unassisted ordinary Easy story starts. Both arms begin with two spare '
                           'lives and two Bombs. Primary endpoint is Stage 2 from this opening. Record remaining stock, '
                           'native stage clocks, terminal image, action trace and effective casts. Controller seeds match '
                           'within host and lane. Original RNG can differ. Engineering seeds 1551301-03 stay excluded.')
    protocol = root / 'camera-protocol.json'
    freeze(protocol, plan)
    selection = root / 'suffix-selection.json'
    while not selection.is_file():
        time.sleep(15)
    attempts_received = json.loads(selection.read_bytes())['attempts']
    if (len(attempts_received) != len(attempts)
            or sorted(r['seed'] for r in attempts_received) != attempts):
        raise ValueError('Incomplete final-spell acquisition denominator')
    accepted = []
    audit = []
    for row in sorted(attempts_received, key=lambda r: r['seed']):
        checked_file(root, row, 'evidence')
        if not isinstance(row['eligible'], bool) or not row['reason']:
            raise ValueError('Missing independent eligibility or reason')
        if not row['eligible']:
            continue
        path = checked_file(root, row, 'receipt')
        expected = root / 'native13' / f'train-merged-sft-{row["seed"]}' / 'receipt.json'
        record = json.loads(path.read_bytes())
        if (path != expected or record['role'] != 'train' or record['seed'] != row['seed']
                or record['game_sha256'] != plan['native_game_sha256']):
            raise ValueError('Unregistered supervised-policy suffix')
        native = verify(path.parent, actor, prefix=False, expected_rank=0)
        audit.append(native)
        accepted.append(row['seed'])
    freeze(root / 'verified-suffixes.json', dict(attempts=len(attempts), included=accepted, checks=audit))
    if not accepted:
        print('No eligible suffix in the registered acquisition cohort', flush=True)
        return
    fit = {**plan, 'train_seeds': {'13': accepted}, 'study_sha256': digest(protocol),
           'selection_sha256': digest(selection), 'native_training_protocol_sha256': digest(root / 'native13/protocol.json')}
    fit_protocol = root / 'fit-protocol.json'
    freeze(fit_protocol, fit)
    subprocess.run([sys.executable, '-u', '-m', 'scripts.run_registered_camera_update', '--root', root,
                    '--parent', model, '--from-sft', '--protocol', fit_protocol], check=True)
    subprocess.run([sys.executable, '-m', 'scripts.render_cross_title_credits', '--difficulty', 'Easy',
                    '--root', root, '--out', root / 'evaluation/easy-ordinary-stock.mp4'], check=True)


if __name__ == '__main__':
    main()
