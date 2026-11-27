"""Isolate new stock from a continuous original TH17 process while preserving the source."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def crop_new_credit(current, previous):
    """Only accept an exact previous exhausted state followed by a fresh stock at tick zero."""
    source = current / 'native.tsv'
    original = current / 'native-full-process.tsv'
    if not original.exists():
        shutil.copy2(source, original)
    with original.open(newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    with (previous / 'native.tsv').open(newline='') as f:
        final = list(csv.DictReader(f, delimiter='\t'))[-1]
    with (current / 'decisions.tsv').open(newline='') as f:
        action = next(csv.DictReader(f, delimiter='\t'))
    if int(final['spares']) != -1:
        raise ValueError('Previous native credit has live stock')
    boundary = next((i for i, row in enumerate(rows) if int(row['spares']) >= 0), None)
    if boundary is None or any(row[k] != final[k] for row in rows[:boundary] for k in final if k != 'ms'):
        raise ValueError('Unrelated or unclassified native state precedes new stock')
    first = rows[boundary]
    identity = ('stage','start_stage','tick','score','spares','bombs','continues')
    if tuple(int(first[k]) for k in identity) != (1,1,0,0,2,3,0):
        raise ValueError('Missing fresh native Stage 1 stock')
    if (int(action['action_t']),int(action['stage']),int(action['spares_before']),int(action['bombs_before'])) != (0,1,2,3):
        raise ValueError('First learned action did not start with full playable stock')
    if not any(int(r['power']) == 100 and int(r['tick']) <= int(action['tick']) for r in rows[boundary:]):
        raise ValueError('New original stock never initialized before first input')
    lines = original.read_bytes().splitlines(keepends=True)
    cropped = source.with_suffix('.tmp')
    cropped.write_bytes(lines[0] + b''.join(lines[boundary+1:]))
    cropped.replace(source)
    proof = dict(full_stream_sha256=digest(original), previous_credit_native_sha256=digest(previous/'native.tsv'),
        episode_native_sha256=digest(source), full_rows=len(rows), excluded_exhausted_rows=boundary,
        previous_final_native_row=final, new_first_native_row=first, first_action_native_tick=int(action['tick']))
    (current/'warm-native-boundary.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof
