"""Measure how adding TH11 changes imitation on the same players in all five games."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

from scripts.checkpoint_storage import digest


def summarize(replays, files, event):
    by_player = defaultdict(list)
    for name in files:
        replay = replays[name]
        by_player[replay['group']].append(replay['aligned'][event])
    players = {}
    for player, stages in by_player.items():
        n = sum(s['n'] for s in stages)
        if not n:
            continue
        players[player] = dict(n=n, correct=sum(s['correct'] for s in stages),
                               nll=sum(s['n'] * s['movement_nll'] for s in stages) / n)
    n = sum(v['n'] for v in players.values())
    correct = sum(v['correct'] for v in players.values())
    return dict(n=n, correct=correct, accuracy=correct/n, equal_player_nll=mean(v['nll'] for v in players.values()),
                by_player=players)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    source = args.study / 'development/held.json'
    fits = json.loads(source.read_bytes())
    files = sorted(fits['four']['replays'])
    if set(files) != set(fits['five']['replays']) or len(files) != 19:
        raise ValueError('Expected the same nineteen held original stages')
    games = sorted({f[:4].upper() for f in files})
    if games != ['TH10', 'TH11', 'TH12', 'TH13', 'TH17']:
        raise ValueError('Unexpected game roster')
    output = {}
    lines = ['# Effect of including TH11 on each game', '',
             'Both camera policies trained for 192,000 sampled windows. Each row compares predictions on the same original human screens. Direction and turn accuracy pool decisions within a game. The movement negative log likelihood weights each player equally within that game. Lower loss is better.', '',
             '| Game | Players | Stages | Correct movement, four → five | Correct turns, four → five | Movement loss, four → five |',
             '|---|---:|---:|---:|---:|---:|']
    for game in games:
        cohort = [f for f in files if f.startswith(game.lower())]
        entry = {}
        for label in ('four', 'five'):
            entry[label] = {event:summarize(fits[label]['replays'], cohort, event) for event in ('all', 'turn', 'start', 'stop')}
        for event in entry['four']:
            a, b = (entry[label][event] for label in ('four', 'five'))
            if {k:v['n'] for k,v in a['by_player'].items()} != {k:v['n'] for k,v in b['by_player'].items()}:
                raise ValueError('Unmatched held players or action windows')
        output[game] = entry
        a,b = (entry[label]['all'] for label in ('four', 'five'))
        u,v = (entry[label]['turn'] for label in ('four', 'five'))
        lines.append(f"| {game} | {len(a['by_player'])} | {len(cohort)} | {a['correct']:,} → {b['correct']:,} / {a['n']:,} | {u['correct']:,} → {v['correct']:,} / {u['n']:,} | {a['equal_player_nll']:.3f} → {b['equal_player_nll']:.3f} |")
    lines.extend(['', 'TH11 contributes a single development player. Several players contributed multiple stages. The four earlier games have appeared in development comparisons throughout the project. Adding TH11 also redistributes the fixed sampling budget across games and difficulties. These observations describe expert action prediction. Ordinary campaign progress is measured separately.', '',
                  '[Player-level results](per-game-transfer.json) · [Matched checkpoints and source allocations](../README.md)'])
    target = source.parent
    (target/'per-game-transfer.json').write_text(json.dumps(dict(source_sha256=digest(source), games=output), indent=2)+'\n', encoding='utf8')
    (target/'per-game-transfer.md').write_text('\n'.join(lines)+'\n', encoding='utf8')


if __name__=='__main__':main()
