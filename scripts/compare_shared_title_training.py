"""Compare one four-title policy to four specialists under a fixed total SFT budget."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from scripts.checkpoint_storage import digest


def read(path):
    return json.loads(path.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'reference', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args()
    d, out = a.data.resolve(), a.out.resolve()
    study = d/'ranked-multigame-sft'
    joint = study/'four-title'
    base = read(study/'design.json')
    parent = d/base['parent']
    shared = read(joint/'report.json')
    if shared['encoder'] != digest(parent/'camera-policy.pt') or digest(joint/'policy.ts') != shared['torchscript_sha256']:
        raise ValueError('Shared policy provenance differs')
    split = read(d/base['base_split'])['split']
    paths = {r['sha256']: d/r['file'] for r in split['train']}
    paths.update({r['archive_sha256']: d/r['path'] for r in read(d/base['expanded_manifest'])['records']})
    for r in read(d/'human-retail17/train-qualified-02-stages-frozen.json')['qualified']:
        paths[r['archive_sha256']] = d/r['path']
    train, held = {}, {}
    for r in shared['train']:
        f = paths[r['sha256']]
        if digest(f) != r['sha256']:
            raise ValueError('Shared human training pixels changed')
        train.setdefault(r['title'], []).append(f)
    holdout = [(d/r['file'], r['sha256']) for r in split['held']]
    holdout.extend((d/r['path'], r['archive_sha256']) for r in read(d/'human-retail17/held-qualified-02-stages-frozen.json')['qualified'])
    for f, sha in holdout:
        if digest(f) != sha:
            raise ValueError('Held human pixels changed')
        title = read(f.with_suffix('.json'))['title']
        held.setdefault(title, []).append(f)
    if set(train) != set(held) or len(train) != 4:
        raise ValueError('Expected four matched titles')
    plan = dict(parent=digest(parent/'camera-policy.pt'), shared=digest(joint/'policy.ts'),
                updates_per_specialist=750, shared_updates=shared['steps'], batch_size=64,
                question='Allocate 3000 total SFT updates to a single joint policy or four per-title specialists. All use the same 38-pilot initialization, native human photo inputs, two high difficulties and original held pilots. Specialists receive 48000 draws per title; joint realized draws are reported below. This tests training allocation under one optimization seed. The shared parent already has prior exposure to TH10, TH12 and TH13.',
                shared_draws=shared['observed_sampled_windows_by_game_rank'],
                train={t:[digest(f) for f in fs] for t,fs in train.items()},
                evaluation='Repeated development pilots. Per-title matched movement accuracy, changed-direction accuracy, NLL and Bomb onset measures. The observed six retail prior pilots have mixed difficulty. No game progress inference.',
                held={t:[digest(f) for f in fs] for t,fs in held.items()})
    out.mkdir(parents=True, exist_ok=True)
    registration = out/'protocol.json'
    if registration.exists() and read(registration) != plan:
        raise ValueError('Registered shared/specialist comparison differs')
    registration.write_text(json.dumps(plan, indent=2)+'\n')
    for title in sorted(train):
        arm = out/(title+'-only')
        if not (arm/'report.json').exists():
            subprocess.run([sys.executable, '-u', '-m', 'scripts.train_human_camera_policy',
                            '--encoder', str(parent/'camera-policy.pt'), '--reference', str(a.reference.resolve()),
                            '--out', str(arm), '--steps', '750', '--cpu-replay', '--natural-movement-frequency',
                            '--balance-titles', '--balance-difficulties', '--train', *map(str, train[title]),
                            '--held', *map(str, held[title])], check=True)
        report = read(arm/'report.json')
        if (report['encoder'] != plan['parent'] or report['steps'] != 750 or
            {r['sha256'] for r in report['train']} != set(plan['train'][title])):
            raise ValueError('Specialist initialization, update count or game data differs')
        scores = out/'evaluation'/title
        for module, dest in [('compare_human_camera_policies', scores), ('score_human_bomb_onsets', scores/'bomb-onsets.json')]:
            subprocess.run([sys.executable, '-m', 'scripts.'+module, '--models', str(joint), str(arm),
                            '--held', *map(str, held[title]), '--out', str(dest)], check=True)
        print('Verified', title, digest(arm/'policy.ts'), flush=True)


if __name__ == '__main__':
    main()
