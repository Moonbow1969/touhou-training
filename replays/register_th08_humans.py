"""Freeze and decode distinct TH08 Hard/Lunatic human story replays across four teams."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from html import unescape
import json
from pathlib import Path
import re
import unicodedata
import urllib.request

import numpy as np
from replays.build_corpus import ARCHIVE, EXCLUDE
from replays.th08_inputs import TEAMS, full_actions, read
from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze


def register(index):
    cells = defaultdict(list)
    for row in re.findall(r'<tr[^>]*>(.*?)</tr>', index.read_text(encoding='utf8'), re.S):
        link = re.search(r'href="(/replays/royalflare/th08/th8_[a-zA-Z0-9_]+\.rpy)"', row)
        fields = [unescape(re.sub('<[^>]*>', '', f)).strip() for f in re.findall(r'<td[^>]*>(.*?)</td>', row, re.S)]
        if not link or len(fields) != 9 or fields[4] not in ('HardFinalB', 'LunaticFinalB'):
            continue
        rank = fields[4].removesuffix('FinalB')
        team = ' '.join(fields[3].split())
        name = unicodedata.normalize('NFKC', fields[6]).casefold().strip()
        if team not in TEAMS or not name or EXCLUDE.search(fields[7]):
            continue
        cells[rank, team].append(dict(source=ARCHIVE + link[1], author=name, credit=fields[6], rank=rank,
                                      team=team, score=int(fields[1].replace(',', ''))))
    used = set()
    selected = []
    for rank in ('Hard', 'Lunatic'):
        for team in TEAMS:
            rows = sorted(cells[rank, team], key=lambda r: sha256(r['source'].encode()).digest())
            count = 0
            for r in rows:
                if r['author'] in used:
                    continue
                used.add(r['author'])
                selected.append(dict(r, split='train' if count < 6 else 'held'))
                count += 1
                if count == 8:
                    break
            if count != 8:
                raise ValueError(f'Missing independent credited authors in {rank} {team}')
    return selected


def acquire(row, cache):
    path = cache / row['source'].rsplit('/', 1)[-1]
    if not path.is_file():
        with urllib.request.urlopen(row['source'], timeout=30) as stream:
            payload = stream.read(2_000_001)
        if len(payload) > 2_000_000:
            raise ValueError('Replay size exceeds supported format')
        path.write_bytes(payload)
    meta, stages = read(path)
    expected = {'1', '2', '3', '5', '6B'}
    if (meta['rank'] != row['rank'] or meta['shot'] != TEAMS.index(row['team']) or meta['finish'] != 'Clear'
            or meta['score'] != row['score'] or str(meta['score']) != meta['user_score']
            or set(stages) not in (expected | {'4A'}, expected | {'4B'})):
        raise ValueError(f'Original rank, shot, score, clear or stage route mismatch: {meta}')
    counts = {}
    arrays = {}
    for stage, words in stages.items():
        actions = full_actions(words)
        if not len(actions) or len(actions) != len(words) // 4:
            raise ValueError('Incomplete four-tick source')
        arrays[stage] = actions.astype('u1')
        counts[stage] = dict(recorded_input_words=len(words), decisions=len(actions),
                             key_stream_sha256=sha256(words.astype('<u2').tobytes()).hexdigest(),
                             action_sha256=sha256(arrays[stage].tobytes()).hexdigest())
    np.savez_compressed(cache / (path.stem + '-actions.npz'), **arrays)
    return dict(row, replay=path.name, replay_sha256=digest(path), metadata=meta, stages=counts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('index', 'out', 'cache'):
        p.add_argument('--' + name, required=True, type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    a.cache.mkdir(parents=True, exist_ok=True)
    selected = register(a.index)
    protocol = a.out / 'protocol.json'
    freeze(protocol, dict(index_sha256=digest(a.index), source='https://maribelhearn.com/royalflare/th08',
                          format_reference='https://github.com/kauri-off/th08-rs/blob/48b29c5643d476884744529a42b4a5de1c00b60c/specs/15_files_replay_score.md',
                          study='Standalone TH08 action-only source cohort. Six training and two held credited names '
                                'per team and rank. Names are disjoint across all cells within this registration. '
                                'No claim of independence from historical models. Freeze before reading binary records. '
                                'Excluded files stay visible, without substitutes. Images and online outcomes are absent.',
                          records=selected))

    def collect(row):
        try:
            return acquire(row, a.cache), None
        except (OSError, ValueError, UnicodeError) as exc:
            return None, dict(row, reason=str(exc))

    with ThreadPoolExecutor(3) as pool:
        results = list(pool.map(collect, selected))
    good = [ok for ok, _ in results if ok is not None]
    excluded = [error for _, error in results if error is not None]
    totals = defaultdict(lambda: defaultdict(int))
    for row in good:
        cell = totals[row['rank'] + '-' + row['split']]
        cell['replays'] += 1
        for s in row['stages'].values():
            cell['recorded_input_words'] += s['recorded_input_words']
            cell['four_tick_decisions'] += s['decisions']
    if len({row['replay_sha256'] for row in good}) != len(good):
        raise ValueError('Duplicate original bytes across source assignments')
    (a.out / 'source-manifest.json').write_text(json.dumps(dict(protocol_sha256=digest(protocol),
        roster=len(selected), qualified=good, excluded=excluded, totals=totals), ensure_ascii=False, indent=2)
        + '\n', encoding='utf8')
    print(len(good), 'qualified', len(excluded), 'excluded', json.dumps(totals), flush=True)


if __name__ == '__main__':
    main()
