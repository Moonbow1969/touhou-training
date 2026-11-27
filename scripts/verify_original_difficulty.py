"""Verify a registered higher difficulty directly against a complete original-game trace."""
import csv
import gzip

import numpy as np

from scripts.checkpoint_storage import digest


def verify(title, receipt_file, receipt, expected):
    if expected not in (2,3) or receipt.get('difficulty')!=expected:
        raise ValueError(f'Unexpected original difficulty for {receipt_file}')
    folder=receipt_file.parent
    if title==11:
        from scripts.qualified_th11_stage2 import read_credit
        row=read_credit(folder)
        return dict(difficulty=expected,difficulty_verified_native_samples=receipt['native_samples'],
                    difficulty_evidence_sha256=row['trajectory_sha256'])
    if title==17:
        from scripts.verify_th17_original_credit import inspect
        result=inspect(receipt_file,receipt,expected)
        return {k:result[k] for k in ('difficulty','difficulty_verified_native_samples','difficulty_evidence_sha256')}
    if title==10:
        source=folder/'native.csv.gz'
        with gzip.open(source,'rt') as stream:
            values=[int(r['difficulty']) for r in csv.DictReader(stream)]
        if digest(source)!=receipt['stage2_evidence']['native_trace_sha256']:
            raise ValueError('Changed TH10 stage/rank stream')
    elif title==13:
        source=folder/'steps.bin'
        raw=np.fromfile(source,dtype='<f4').reshape(-1,143)
        values=raw[:,141]
        if len(raw)!=receipt['steps'] or digest(source)!=receipt['native_difficulty_trace_sha256']:
            raise ValueError('Changed or incomplete TH13 rank stream')
    elif title==12:
        source=folder.parent/'raw'/receipt['episode']/'difficulty-native.tsv'
        with source.open(encoding='utf8') as stream:
            rows=list(csv.DictReader(stream,delimiter='\t'))
        active=[r for r in rows if int(r['stage'])>=1 and int(r['mode'])==0 and int(r['frame'])>=0]
        values=[int(r['rank']) for r in active]
        if not active or int(active[0]['spares'])!=2 or int(active[-1]['spares'])!=-1:
            raise ValueError('TH12 difficulty trace omits the opening or terminal')
        if any(int(r['rank'])!=int(r['recorded_rank']) or int(r['character'])!=0 or int(r['shot'])!=0 for r in active):
            raise ValueError('Live TH12 difficulty disagrees with the original recorder header')
        if digest(source)!=receipt['native_difficulty_trace_sha256']:
            raise ValueError('Changed TH12 original difficulty stream')
    else:raise ValueError(title)
    if not len(values) or len(values)!=receipt['difficulty_verified_native_samples'] or not np.all(np.asarray(values)==expected):
        raise ValueError(f'Original difficulty or evidence length mismatch at {receipt_file}')
    return dict(difficulty=expected,difficulty_verified_native_samples=len(values),difficulty_evidence_sha256=digest(source))
