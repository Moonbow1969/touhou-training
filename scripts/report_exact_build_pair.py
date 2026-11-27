"""Measure the effect of exact-build human stages at fixed fine-tuning budget."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def summarize(arm, bombs):
    players = defaultdict(list)
    for name, stage in arm['replays'].items():
        if bombs[name]['camera_sha256'] != stage['sha256']:
            raise ValueError('Movement and Bomb metrics came from different stage cameras')
        players[stage['group']].append((stage['aligned'], bombs[name]))

    def group_metrics(rows):
        result = {}
        for condition in ('all', 'change', 'start', 'turn'):
            counts = [m[condition] for m, _ in rows]
            n = sum(m['n'] for m in counts)
            result[condition] = {
                'n': n, 'accuracy': sum(m['correct'] for m in counts) / n if n else None,
                'nll': sum(m['movement_nll'] * m['n'] for m in counts if m['n']) / n if n else None,
            }
        bomb_rows = [b['aligned'] for _, b in rows]
        result['bomb'] = {k: sum(b[k] for b in bomb_rows) for k in (
            'onset_windows', 'top_five_percent_onsets', 'top_five_percent_flagged', 'rows')}
        return result

    return {'players': {p: group_metrics(rows) for p, rows in players.items()},
            'pooled': group_metrics([r for rows in players.values() for r in rows])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    root = parser.parse_args().evaluation
    names = ['retail-held-six', 'TH12-trial', 'TH13-trial']
    arms = ('retail-rehearsal', 'trial-augmented')
    results = {}
    for name in names:
        movement = json.loads((root/name/'held.json').read_bytes())
        bombs = json.loads((root/name/'bomb-onsets.json').read_bytes())
        old, new = (movement[a]['replays'] for a in arms)
        if set(old) != set(new) or any(old[s]['sha256'] != new[s]['sha256'] for s in old):
            raise ValueError('Unpaired camera observations')
        results[name] = {arm: summarize(movement[arm], bombs[arm]) for arm in arms}
    (root/'matched-human-metrics.json').write_text(json.dumps(results, indent=2)+'\n')

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), constrained_layout=True)
    x = np.arange(len(names))
    for ax, cond, label in zip(axes, ('all', 'change', 'start'),
                               ('All movements', 'Direction changed', 'Started from rest')):
        for j, arm in enumerate(arms):
            values = [100 * results[n][arm]['pooled'][cond]['accuracy'] for n in names]
            bars = ax.bar(x + (j-.5)*.38, values, .38,
                          color=('#677d8d','#17795d')[j],
                          label=('Retail rehearsal', 'With exact-build replays')[j])
            ax.bar_label(bars, fmt='%.1f', padding=2, fontsize=8)
        ax.set_xticks(x, ['Retail\n6 pilots', 'TH12 trial\n2 pilots', 'TH13 trial\n2 pilots'])
        ax.set_title(label)
        ax.set_ylim(0, max(max(b.get_height() for b in c) for c in ax.containers) * 1.22)
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Held expert movement accuracy (%)')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=2)
    fig.savefig(root/'matched-human-movement.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
