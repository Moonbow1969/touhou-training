"""Train a fixed-camera policy after a verified new-player milestone seals."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

from scripts.checkpoint_storage import digest


def verified_file(root, name, checksum):
    p = (root / name).resolve()
    p.relative_to(root)
    if digest(p) != checksum:
        raise ValueError(f'Changed replay shard {name}')
    return p


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'base-protocol', 'milestone', 'encoder', 'reference', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--architecture',choices=('cnn','spatial-attention'),default='spatial-attention')
    args = p.parse_args()
    root = args.root.resolve()
    fixed = json.loads(args.base_protocol.read_bytes())['split']
    milestone = json.loads(args.milestone.read_bytes())
    extra = milestone['records']
    if milestone['split'] != 'train' or len(extra) != milestone['unique_new_pilots']:
        raise ValueError('Incomplete milestone')
    old_train = {r['player'] for r in fixed['train']}
    old_held = {r['player'] for r in fixed['held']}
    added = {r['player_group'] for r in extra}
    if len(added) != len(extra) or added & (old_train | old_held) or old_train & old_held:
        raise ValueError('Repeated or held player')
    if any(r['split'] != 'train' for r in extra):
        raise ValueError('A new training shard has the wrong split')
    if len({r['replay_sha256'] for r in extra}) != len(extra):
        raise ValueError('Repeated replay in milestone')
    if milestone['union_decisions'] != milestone['base_train_decisions'] + sum(r['sft_four_tick_decisions'] for r in extra):
        raise ValueError('Milestone counts disagree')
    train = [verified_file(root, r['file'], r['sha256']) for r in fixed['train']]
    held = [verified_file(root, r['file'], r['sha256']) for r in fixed['held']]
    train += [verified_file(root, r['path'], r['archive_sha256']) for r in extra]
    if len(set(train + held)) != len(train) + len(held):
        raise ValueError('Repeated camera source in join')
    steps = math.ceil(12 * milestone['union_decisions'] / 64)
    args.out.mkdir(parents=True, exist_ok=True)
    train_dir = args.out / 'policy'
    receipt = dict(base_protocol_sha256=digest(args.base_protocol), milestone_sha256=digest(args.milestone),
                   train_players=len(old_train | added), held_players=len(old_held), train_windows=milestone['union_decisions'],
                   optimizer_steps=steps, presentations=64*steps, initialization_sha256=digest(args.encoder),
                   sampling='Twelve corpus-equivalent presentations under the original title-balanced, transition-enriched sampler.',
                   actor=args.architecture, player_location_weight=.15)
    plan_file = args.out / 'protocol.json'
    if plan_file.exists() and json.loads(plan_file.read_bytes()) != receipt:
        raise ValueError('Existing training registration differs')
    plan_file.write_text(json.dumps(receipt, indent=2)+'\n')
    def run(module, *arguments):
        subprocess.run([sys.executable, '-u', '-m', 'scripts.'+module, *map(str, arguments)], check=True)
    if not (train_dir / 'report.json').exists():
        run('train_human_camera_policy', '--encoder', args.encoder, '--reference', args.reference,
            '--train', *train, '--held', *held, '--out', train_dir, '--steps', steps, '--natural-movement-frequency',
            '--balance-titles', '--player-location-weight', .15, '--cpu-replay',
            *(['--spatial-attention'] if args.architecture=='spatial-attention' else []))
    actual = json.loads((train_dir / 'report.json').read_bytes())
    if sum(r['decisions'] for r in actual['train']) != receipt['train_windows'] or actual['steps'] != steps:
        raise ValueError('Actual training windows or optimizer budget differs')
    run('compare_human_camera_policies', '--models', train_dir, '--held', *held, '--out', args.out/'held-movement')
    run('score_human_bomb_onsets', '--models', train_dir, '--held', *held, '--out', args.out/'held-onsets.json')


if __name__ == '__main__':
    main()
