"""Compute the action information lost by holding the original replay's first command."""
from collections import defaultdict
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np

work=Path(__file__).resolve().parent
data=work.parent/'outputs/touhou-vision-261126'
root=data/'th10-retail-normal'
study=root/'normal-sensor-diagnostic'
fit=json.loads((root/'normal-sft-protocol.json').read_bytes())
sources=[dict(v,role='demonstration_train') for v in fit['train']]
for stage in (1,2):
    file=f'human-cross-title-v2/held/th10-gensokyo-2749-s{stage}.npz'
    sidecar=json.loads((data/file).with_suffix('.json').read_bytes())
    sources.append(dict(file=file,sha256=sidecar['archive_sha256'],player='duckator',shot='ReimuA',role='held'))


def rows(path):
    with np.load(path,allow_pickle=False) as z:
        clock=z['native_frame'].astype('i8')
        chunks=z['input_chunk'].astype('i8')
        stable=z['stable'].astype(bool) if 'stable' in z else np.ones(len(clock),bool)
    if chunks.shape!=(len(clock),4) or np.any(np.diff(clock)<=0):
        raise ValueError(f'Bad original action intervals {path.name}')
    index=[]
    at=0
    while at+1<len(clock):
        if clock[at+1]-clock[at]==4 and stable[at] and stable[at+1]:
            index.append(at)
            at+=2
        else:
            at+=1
    if not index:raise ValueError('Original replay has no eight-tick continuous blocks')
    index=np.asarray(index)
    return np.concatenate([chunks[index],chunks[index+1]],axis=1)


entries=[]
for source in sources:
    path=data/source['file']
    with path.open('rb') as f:
        if file_digest(f,'sha256').hexdigest()!=source['sha256']:raise ValueError('Changed original replay shard')
    seq=rows(path)
    result={}
    for hold in (2,4,8):
        result[str(hold)]={}
        for key,mask in (('direction',240),('focus',4),('shoot',1),('bomb',2),('all_seven_keys',247)):
            words=(seq&mask).reshape(-1,hold)
            mismatch=int(np.sum(words!=words[:,:1]))
            optimal=int(sum(hold-np.unique(window,return_counts=True)[1].max() for window in words))
            result[str(hold)][key]=dict(mismatch=mismatch,ticks=words.size,optimistic_minimum=optimal)
    entries.append(dict(source,metrics=result))
groups=defaultdict(list)
for entry in entries:
    groups[(entry['role'],entry['player'])].append(entry)
summary={}
for name,match in [('ReimuA_train',lambda r:r['role']=='demonstration_train' and r['shot']=='ReimuA'),
                   ('all_Normal_train',lambda r:r['role']=='demonstration_train'),
                   ('held_ReimuA',lambda r:r['role']=='held')]:
    subset=[r for r in entries if match(r)]
    out={}
    for hold in (2,4,8):
        columns=[v['metrics'][str(hold)]['direction'] for v in subset]
        total=sum(c['ticks'] for c in columns)
        out[str(hold)]=dict(ticks=total,mismatch=sum(c['mismatch'] for c in columns),
                            optimistic_minimum=sum(c['optimistic_minimum'] for c in columns))
    summary[name]=dict(players=len({r['player'] for r in subset}),stages=len(subset),direction=out)
result=dict(procedure='Choose non-overlapping stable adjacent four-tick replay chunks, treating every hold length on the same eight original input words. No resimulation. Optimistic bound uses the most common command over each future window. Training demographics are descriptive. The single held pilot did not participate in this SFT.',
            groups=summary,entries=entries)
(study/'human-recorded-control-bandwidth.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
for name,group in summary.items():
    print(name,group['players'],[(k,round(100*c['mismatch']/c['ticks'],3),c['ticks']) for k,c in group['direction'].items()])
