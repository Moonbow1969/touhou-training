"""Build a bounded source envelope after the original closed, with full native time and minimal original HUD evidence."""
import argparse
import ast
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import tarfile
import zipfile


def sha(path):
    with path.open('rb')as f:
        d=hashlib.sha256()
        for b in iter(lambda:f.read(1024*1024),b''):d.update(b)
    return d.hexdigest()

def bytes_sha(b):return hashlib.sha256(b).hexdigest()

def hud_slices(npz,indices):
    selected=set(indices)
    stock=bytearray();rank=bytearray()
    with zipfile.ZipFile(npz)as archive,archive.open('hud.npy')as f:
        assert f.read(6)==b'\x93NUMPY';version=f.read(2);sz=2 if version[0]==1 else 4
        header=ast.literal_eval(f.read(int.from_bytes(f.read(sz),'little')).decode('latin-1').strip())
        n,y,x,c=header['shape'];assert (y,x,c)==(204,206,3) and header['descr']=='|u1' and not header['fortran_order'] and max(indices)<n
        for i in range(n):
            image=f.read(y*x*c)
            if i not in selected:continue
            assert len(image)==y*x*c
            for j in range(68,97):stock.extend(image[(j*x+60)*3:(j*x+185)*3])
            for j in range(170,203):rank.extend(image[(j*x+104)*3:(j*x+205)*3])
    assert len(stock)==len(indices)*29*125*3 and len(rank)==len(indices)*33*101*3
    return bytes(stock),bytes(rank)

def compact(raw,destination):
    origin=json.loads((raw/'receipt.json').read_bytes());result=json.loads((raw/'fragment-receipt.json').read_bytes())
    release = json.loads((raw/'branch-release.json').read_bytes())
    provenance = json.loads((raw/'branch-provenance.json').read_bytes())
    registration = json.loads((raw/'branch-protocol.json').read_bytes())
    enrolled = [entry for entry in registration['rows'] if entry['seed'] == result['seed']]
    assert len(enrolled) == 1 and result['role'] == release['role'] == enrolled[0]['role'] == 'held'
    assert result['seed'] == release['controller_seed'] and result['seed'] in range(80965501,80965541)
    assert release['arm'] == enrolled[0]['arm'] and release['arm'] in ('lambda1-parent','mixed-ppo')
    assert result['policy_sha256'] == release['policy_sha256'] == enrolled[0]['policy_sha256']
    assert result['source_policy_sha256'] == origin['policy_sha256'] == release['source_policy_sha256']
    assert (result['policy_sha256'] == result['source_policy_sha256']) == (release['arm'] == 'lambda1-parent')
    assert provenance['release_sha256'] == sha(raw/'branch-release.json')
    assert result['protocol_sha256'] == release['protocol_sha256'] == sha(raw/'branch-protocol.json')
    assert sha(raw/'behavior.npz')==origin['behavior_sha256'] and sha(raw/'native-observers.npz')==origin['native_observers_sha256'] and sha(raw/'fragment.npz')==result['sha256']
    assert sha(raw/'timing.jsonl')==origin['timing_sha256']
    rows=[json.loads(x)for x in (raw/'timing.jsonl').read_text().splitlines()]
    live=[i for i,r in enumerate(rows)if r['phase']=='policy'][result['offset']:]
    assert len(live)==result['decisions'] and rows[-1]['phase'] in ('fragment_boundary','terminal')
    ids=live+[len(rows)-1];stock,rank=hud_slices(raw/'native-observers.npz',ids)
    source={name:(raw/name).read_bytes() for name in result['native_source_sha256'] if name not in ('behavior.npz','clock.tsv')}
    for name,b in source.items():assert bytes_sha(b)==result['native_source_sha256'][name],name
    full_clock=(raw/'clock.tsv').read_bytes()
    expected=result['native_source_sha256']['clock.tsv'];signed=full_clock;tail=b''
    if bytes_sha(signed)!=expected:
        h=hashlib.sha256();count=0
        for line in io.BytesIO(signed):
            h.update(line);count+=len(line)
            if h.hexdigest()==expected:break
        if h.hexdigest()!=expected:raise ValueError('Native source-clock bytes changed before registered boundary')
        signed,tail=full_clock[:count],full_clock[count:]
        assert all(int(l.split()[0])>rows[-1]['monotonic_ns']for l in tail.splitlines())
    source['clock.tsv']=signed
    if tail:source['clock-after-signature.tsv']=tail
    for filename in ['fragment-receipt.json','branch-provenance.json','branch-release.json','branch-protocol.json','vm-anchor.json','vm-policy-anchor.json','late-registration.json','late-deployed-binaries.sha256','live-binaries.sha256','build.sha256','ordinary-proof.json','source.sha256','native-archive-bracket-status.tsv','post-credit-settled.png','post-credit-confirm.png']:
        if (raw/filename).is_file():source[filename]=(raw/filename).read_bytes()
    for path in raw.glob('native-record-stage-*'):
        if path.suffix in ('.bin','.partial'):source[path.name]=path.read_bytes()
    source['stock-rgb.ubyte']=stock;source['rank-rgb.ubyte']=rank
    evidence=dict(seed=result['seed'],native_stage=result['stage'],verified_observer_full_sha256=origin['native_observers_sha256'],
        full_behavior_sha256=origin['behavior_sha256'],fragment_sha256=result['sha256'],raw_full_clock_sha256=bytes_sha(full_clock),
        signed_clock_sha256=bytes_sha(signed),signed_clock_bytes=len(signed),post_boundary_appended_bytes=len(tail),source_timing_row_indices=ids,
        stock=dict(roi_xyxy=[60,68,185,97],shape=[len(ids),29,125,3]),normal_rank=dict(roi_xyxy=[104,170,205,203],shape=[len(ids),33,101,3]),
        source_sha256={k:bytes_sha(v)for k,v in source.items()},full_source_sha256=result['native_source_sha256'],
        scope='Causal original action cameras and next-state camera in separate fragment.npz. Full signed clock and native control witness retained. HUD and rank crops are the exact support read from original source HUD array after checking its original camera-receipt hash.')
    source['compact-seal.json']=(json.dumps(evidence,sort_keys=True)+'\n').encode()
    pending=destination.with_suffix('.pending')
    with pending.open('wb')as out,gzip.GzipFile(filename='',mode='wb',compresslevel=2,fileobj=out,mtime=0)as gz,tarfile.open(fileobj=gz,mode='w|')as ar:
        for name,b in sorted(source.items()):
            if not name or Path(name).name!=name:raise ValueError(name)
            member=tarfile.TarInfo(name);member.size=len(b);member.mode=0o600;ar.addfile(member,io.BytesIO(b))
    os.replace(pending,destination)
    print(json.dumps(dict(seed=result['seed'],compact_sha256=sha(destination),bytes=destination.stat().st_size,fragment_sha256=sha(raw/'fragment.npz'),fragment_bytes=(raw/'fragment.npz').stat().st_size)),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('capture',type=Path);p.add_argument('out',type=Path);a=p.parse_args();compact(a.capture,a.out)
