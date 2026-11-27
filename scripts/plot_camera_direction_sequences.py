"""Publish the measured direction changes on fixed human screens and native policy trajectories."""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--public', required=True, type=Path)
    args = parser.parse_args()
    root = args.data / 'camera-direction-sequences'
    dest = args.public / root.name
    dest.mkdir(exist_ok=True, parents=True)
    for file in root.glob('*.json'):
        shutil.copy2(file, dest / file.name)
    stages = json.loads((root / 'held-human-screens.json').read_bytes())['stages']
    credits = json.loads((root / 'actual-six-title-commands.json').read_bytes())['credits']
    targets = ['expert', 'later-sft', 'later-ppo']
    labels = ['Human', 'Human SFT', 'SFT → PPO']
    colors = ['#423955', '#087f79', '#b14e2f']
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), layout='constrained')
    page = ['# Consecutive movement commands on identical human screens', '',
            'Four previously examined TH17 players supplied eight synchronized Hard and Lunatic stages. The comparison follows the same human screens and the same adjacent four-tick observation pairs. Human directions are observed at camera boundaries. The policies independently sample their learned direction distribution at each boundary. The table uses their exact expected counts.', '',
            '| Controller | Adjacent directions | Change direction | Directly reverse | Issue neutral | Eight-command RMS resultant |',
            '|---|---:|---:|---:|---:|---:|']
    summaries = []
    for label, key in zip(labels, targets):
        rows = [v['metrics'][key] for v in stages]
        def total(name):
            return sum(v[name] for v in rows)
        pairs = total('pairs')
        change, reverse, neutral = [100 * total(field) / pairs for field in ('changed', 'opposite', 'stationary')]
        rms = np.sqrt(total('sum_squared_normalized_resultant') / total('eight_command_windows'))
        page.append(f'| {label} | {pairs:,} | {change:.2f}% | {reverse:.2f}% | {neutral:.2f}% | {rms:.3f} |')
        summaries.append((change, reverse, rms))
    for ax, values, label in zip(axes, np.asarray(summaries).T, ['Direction changes (%)', '180° reversals (%)', 'Eight-command RMS resultant']):
        bars = ax.bar(labels, values, color=colors, width=.68)
        ax.bar_label(bars, fmt='%.2f', fontsize=9)
        ax.set_ylim(0, max(values) * 1.19)
        ax.set_title(label, fontsize=11)
        ax.tick_params(axis='x', labelrotation=12)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.2)
    figure.suptitle('Direction sequences on four held TH17 players’ original screens')
    figure.savefig(dest / 'direction-sequences.png', dpi=150)
    plt.close(figure)
    page += ['', 'The eight-command measure sums unit direction vectors (zero for neutral), divides by eight, and reports the root mean square length across windows. Opposing commands cancel. The upper bound is one. Actual travel also depends on focus speed, boundaries, game state and which screen earlier actions produce.', '',
             '![Direction commands on the same TH17 images](direction-sequences.png)', '',
             '## Commands issued during six-game PPO collection', '',
             'Each cell contains 16 ordinary original training starts under the previous six-title policy. Adjacent decisions can cover different native durations across games. These rows use the sampled action log from the policy’s own play. Native key-readback audits are linked separately.', '',
             '| Title | Difficulty | Adjacent commands | Changed direction | Directly reversed |', '|---|---|---:|---:|---:|']
    for title in sorted({v['title'] for v in credits}):
        for rank, name in [(2, 'Hard'), (3, 'Lunatic')]:
            cell = [v for v in credits if v['title'] == title and v['rank'] == rank]
            if len(cell) != 16:
                raise ValueError('Incomplete title and difficulty')
            n = sum(v['pairs'] for v in cell)
            page.append(f'| TH{title:02} | {name} | {n:,} | {100 * sum(v["changed"] for v in cell) / n:.2f}% | {100 * sum(v["opposite"] for v in cell) / n:.2f}% |')
    page += ['', '[Per-stage human evidence](held-human-screens.json) · [All 192 native action sequences](actual-six-title-commands.json) · [TH10, TH11, TH13 readback](../six-game-return-horizon/three-title-original-action-coverage.json) · [TH08, TH12, TH17 readback](../six-game-return-horizon/th08-th12-th17-original-action-coverage.json)']
    (dest / 'README.md').write_text('\n'.join(page) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
