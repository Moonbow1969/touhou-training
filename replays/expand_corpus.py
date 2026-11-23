import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from html import unescape
import json
from pathlib import Path
import re

from .build_corpus import DIFFICULTIES, EXCLUDE, ROOT, player_group, split_for
from .build_multigame_corpus import fetch_first


TRAINING_TITLES = (13, 14, 16, 17)


def candidates(seed):
    groups = defaultdict(deque)
    for title in TRAINING_TITLES:
        index = (ROOT / f"th{title}-index.html").read_text(encoding="utf-8")
        for raw_row in re.findall(r"<tr[^>]*>(.*?)</tr>", index, re.S):
            link = re.search(rf'href="(/replays/royalflare/th{title}/[^"/]+\.rpy)"', raw_row)
            cells = [unescape(re.sub("<[^>]+>", "", cell)).strip()
                     for cell in re.findall(r"<td[^>]*>(.*?)</td>", raw_row, re.S)]
            if not link or len(cells) != 9:
                continue
            shot, rank, comment = cells[3], cells[4], cells[7]
            if rank in DIFFICULTIES and not EXCLUDE.search(comment) and Path(link[1]).name not in seed:
                groups[title, rank, shot].append((title, rank, shot, link[1], comment))
    active = deque(groups.values())
    while active:
        group = active.popleft()
        yield group.popleft()
        if group:
            active.append(group)


def save(path, rows):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(list(rows.values()), indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description='Expand and verify the native-input replay corpus.')
    parser.add_argument('--target', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.target < 1 or not 1 <= args.workers <= 4:
        parser.error('Use a positive target and between one and four download workers.')
    destination = ROOT / 'expanded-corpus.json'
    source = destination if destination.exists() else ROOT / 'multigame-corpus.json'
    rows = {row['name']: row for row in json.loads(source.read_text(encoding='utf-8'))}
    verified_hashes = {row['sha256'] for row in rows.values()}
    pending = deque(candidates(rows))
    print(f'{len(rows)} verified seed runs, {len(pending)} additional archive candidates', flush=True)
    examined = 0
    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        while pending and len(rows) < args.target:
            size = min(24, len(pending), args.target - len(rows))
            for row in workers.map(fetch_first, ([pending.popleft()] for _ in range(size))):
                if not row or row['sha256'] in verified_hashes:
                    continue
                row['player_group'] = player_group(row['player'])
                row['split'] = split_for(row['player'])
                rows[row['name']] = row
                verified_hashes.add(row['sha256'])
            examined += size
            save(destination, rows)
            print(f'{len(rows)} verified after checking {examined} new candidates', flush=True)
    save(destination, rows)
    print(json.dumps({'verified': len(rows), 'titles': Counter(row['title'] for row in rows.values()),
                      'splits': Counter(row['split'] for row in rows.values()),
                      'frames': sum(sum(row['frames_by_stage'].values()) for row in rows.values())}), flush=True)


if __name__ == '__main__':
    main()
