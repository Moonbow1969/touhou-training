"""Register and score two fixed-compute human SFTs on ordinary Hard and Lunatic games."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.checkpoint_storage import digest
from scripts.run_registered_camera_update import wait_th12_coverage
from scripts.score_original_camera_run import records
from scripts.train_five_original_games import freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    data = parser.parse_args().data.resolve()
    root = data / 'controlled-human-scale-original'
    root.mkdir(exist_ok=True)
    broad = data / 'scaled-human-sft/closed-86'
    studies = {'same-device-64': broad / 'same-device-old64', 'same-device-86': broad}
    seeds = {str(t): list(range(1580051 + 100*t, 1580075 + 100*t)) for t in (10, 12)}
    plan = dict(reference_arm='same-device-64', candidate='same-device-86', eval_seeds=seeds,
                rank_by_seed={t: {str(s): 2 if s % 2 else 3 for s in ss} for t, ss in seeds.items()},
                initialization='Same encoder, four action heads, same optimizer, same Momentum RTX 5090 and seed. '
                               'At exactly 44,149 updates compare 77 versus 97 credited training authors.',
                training_registrations={a: digest(p / 'protocol.json') for a, p in studies.items()},
                expected_sources={'same-device-64': 88, 'same-device-86': 110},
                expected_presentations=2825536,
                design='Twelve fresh control seeds per title and difficulty. Two independent original stock games per seed. '
                       'Both arms wait for the same two checkpoint bindings. Keep the physical host, rank and lane within a pair. '
                       'Balance execution order and retain every technical attempt. Original RNG is uncontrolled.',
                endpoints='Primary Stage 2 reached per title and difficulty. Also report remaining spares and Bombs or Power, '
                          'effective casts, and native Stage 1 progress. No geometric guidance, automatic Bomb, or aiming corrections.',
                interpretation='Both games contribute human SFT training data. This experiment measures the effect of expanding '
                               'the human training cohort on closed-loop deployment at a fixed update budget.')
    protocol = root / 'camera-protocol.json'
    freeze(protocol, plan)
    actors = {}
    for arm, study in studies.items():
        folder = study / 'matched-44149/policy'
        files = [folder / name for name in ('policy.ts', 'camera-policy.pt', 'report.json', 'observed-sampling.json')]
        while not all(f.is_file() for f in files):
            time.sleep(12)
        actor, state, report_file, sampling_file = files
        report = json.loads(report_file.read_bytes())
        sampling = json.loads(sampling_file.read_bytes())
        fingerprint = digest(actor)
        source_roster = json.loads((study / 'protocol.json').read_bytes())['train']
        train_hashes = {r['sha256'] for r in report['train']}
        if (report['steps'] != 44149 or sampling['presentations'] != plan['expected_presentations']
                or digest(state) != report['sft'] or fingerprint != report['torchscript_sha256']
                or train_hashes != {r['archive_sha256'] for r in source_roster}
                or len(train_hashes) != plan['expected_sources'][arm]):
            raise ValueError(f'Changed supervised training, source roster or model for {arm}')
        actors[arm] = dict(file=actor.relative_to(data).as_posix(), actor_sha256=fingerprint,
                           state_sha256=digest(state), sampling_sha256=digest(sampling_file),
                           report_sha256=digest(report_file), presentations=sampling['presentations'],
                           unique_seen=sampling['unique_sampled_windows'])
    freeze(root / 'checkpoint-binding.json', dict(registration_sha256=digest(protocol), arms=actors))
    print('Verified matched 5090 actors', actors, flush=True)
    # Both arms must have control coverage through the original fatal tick.
    expected = {(s, a['actor_sha256']) for s in seeds['12'] for a in actors.values()}
    while True:
        found = {}
        for title, path in records(root, True, True):
            if title != 12:
                continue
            row = json.loads(path.read_bytes())
            key = row['seed'], row['policy_sha256']
            if key in expected:
                if key in found:
                    raise ValueError(f'Duplicate registered TH12 original {key}')
                found[key] = path
        if set(found) == expected:
            break
        time.sleep(20)
    wait_th12_coverage(root / 'th12-held-qualified-coverage.json', found.values())
    out = root / 'evaluation'
    reference, candidate = plan['reference_arm'], plan['candidate']
    cmd = [sys.executable, '-m', 'scripts.score_original_camera_run', '--round2', '--root', root,
           '--protocol', protocol, '--parent', data / actors[reference]['file'],
           '--candidate', data / actors[candidate]['file'], '--arm-names', reference, candidate, '--out', out]
    while True:
        scored = subprocess.run(cmd, capture_output=True, text=True)
        if scored.returncode == 0:
            break
        if 'Waiting for verified whole credits' not in scored.stderr:
            raise RuntimeError(scored.stderr)
        time.sleep(20)
    for module, args in [('plot_full_action_results', ['--folder', out, '--candidates', candidate]),
                         ('summarize_four_title_pair', ['--root', root])]:
        subprocess.run([sys.executable, '-m', 'scripts.' + module, *args], check=True)
    for rank in ('Hard', 'Lunatic'):
        subprocess.run([sys.executable, '-m', 'scripts.render_cross_title_credits', '--root', root,
                        '--difficulty', rank, '--out', out / (rank.lower() + '-full-credits.mp4')], check=True)


if __name__ == '__main__':
    main()
