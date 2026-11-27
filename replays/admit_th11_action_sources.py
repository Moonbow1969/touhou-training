"""Admit complete TH11 input replays from the Royalflare archive for original-camera qualification."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

from replays.build_corpus import ARCHIVE, player_group
from replays.build_multigame_corpus import ROOT, fetch_first, select_groups


def identified_source(candidates):
    for candidate in candidates:
        row = fetch_first([candidate])
        if row and player_group(row['player']) not in ('', 'anonymous', 'player', 'no name'):
            return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    groups = list(select_groups(11))
    index = ROOT / 'th11-index.html'
    registration = dict(game='TH11', scope='Training and development candidate. TH15, TH18 and TH19 are held by title.',
                        source=ARCHIVE + '/royalflare/th11', index_sha256=sha256(index.read_bytes()).hexdigest(),
                        selection='First qualifying complete replay per shot and difficulty in the archived index. Up to four candidates per cell.',
                        candidates=[[row[3] for row in g] for g in groups])
    plan = args.out / 'source-registration.json'
    if plan.exists() and json.loads(plan.read_bytes()) != registration:
        raise ValueError('Source registration changed')
    plan.write_text(json.dumps(registration, indent=2) + '\n', encoding='utf8')
    with ThreadPoolExecutor(max_workers=2) as workers:
        rows = [row for row in workers.map(identified_source, groups) if row]
    if len({r['sha256'] for r in rows}) != len(rows) or not rows:
        raise ValueError('Expected distinct complete original replay streams')
    for row in rows:
        row['player_group'] = player_group(row['player'])
        row['four_tick_windows'] = sum(n // 4 for n in row['frames_by_stage'].values())
    manifest = dict(scope='Decoded native input words only. No game pixels, learner updates or agent credits are included. Historical player split must be reconciled against the active camera training roster before any fit.',
                    sources=len(rows), pilots=len({r['player_group'] for r in rows}),
                    difficulty=dict(Counter(r['difficulty'] for r in rows)),
                    native_input_frames=sum(sum(r['frames_by_stage'].values()) for r in rows),
                    four_tick_windows=sum(r['four_tick_windows'] for r in rows), records=rows)
    (args.out / 'action-source-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf8')
    print({k:v for k,v in manifest.items() if k not in ('records','scope')}, flush=True)


if __name__ == '__main__':
    main()
