"""Bind the registered sources, native receipts and publicly verified artifact digests."""
import hashlib
import json
from pathlib import Path
import shutil
import sys


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def index(study):
    protocol = json.loads((study / "protocol.json").read_bytes())
    results = []
    for q in protocol["roles"]:
        for stage in (1, 2):
            folder = study / "stages" / f"{q['id']:02d}-s{stage}"
            r = json.loads((folder / "release.json").read_bytes())
            a = json.loads((folder / "source-audit.json").read_bytes())
            assert q['id'] == r['source_id'] and q['role'] == r['role'] == a['role']
            assert r['stage'] == a['stage'] == stage and r['author'] == q['author'] == a['author']
            assert a['admitted'] and a['original_replay_sha256'] == q['original_sha256']
            assert digest(folder / 'source-audit.json') == r['source_audit_sha256']
            assert a['protocol_sha256'] == digest(study / 'protocol.json')
            assert a['rows'] == a['native_current_equal'] == a['native_previous_equal'] == a['cursor_equal_expected']
            assert digest(folder / 'positions.npz') == a['artifacts_sha256']['positions.npz']
            for k in ('rows','causal_four_tick_pairs','capped_windows','gray_regenerations','excluded_xy_rows'):
                r[k] = a[k]
            results.append(r)
    root = dict(protocol_sha256=digest(study / 'protocol.json'), stages=results)
    for role in ('train','held'):
        selected = [x for x in results if x['role'] == role]
        root[role] = dict(unique_players=len({x['author'] for x in selected}), stages=len(selected),
                          raw_camera_rows=sum(x['rows'] for x in selected),
                          causal_windows=sum(x['causal_four_tick_pairs'] for x in selected),
                          capped_windows=sum(x['capped_windows'] for x in selected))
    root['total_release_bytes'] = sum(x['bytes'] for x in results)
    (study / 'index.json').write_text(json.dumps(root,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(digest(study / 'index.json'), root['train'], root['held'], flush=True)
    return root


if __name__=='__main__':
    index(Path(sys.argv[1]))
