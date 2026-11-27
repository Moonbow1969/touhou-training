"""Distribute small reward corrections and reconstruct exact original-camera training shards."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from scripts.checkpoint_storage import digest


def safe_name(s):
    if Path(s).name!=s or '/' in s or '\\' in s or not s.endswith('.npz'):
        raise ValueError('Invalid training shard filename')
    return s


def rebuild(raw,entry,patch,original):
    if hashlib.sha256(raw).hexdigest()!=original['sha256']:raise ValueError('Changed raw camera shard')
    if entry['title']==13:
        with np.load(io.BytesIO(raw),allow_pickle=False) as z:
            arrays={k:z[k] for k in z}
        with np.load(io.BytesIO(patch),allow_pickle=False) as z:
            if z.files!=['native_ticks_earned','reward']:raise ValueError('Unexpected correction schema')
            arrays['reward_original']=arrays['reward']
            arrays['native_ticks_earned']=z['native_ticks_earned']
            arrays['reward']=z['reward']
        result=io.BytesIO()
        np.savez_compressed(result,**arrays)
        raw=result.getvalue()
    if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('Reconstructed optimizer shard differs')
    return raw


def indexed(rows):
    lookup={safe_name(r['file']):r for r in rows}
    if len(lookup)!=len(rows):raise ValueError('Duplicate training filenames')
    return lookup


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    choices=parser.add_subparsers(dest='mode',required=True)
    pack=choices.add_parser('pack')
    pack.add_argument('--study',type=Path,required=True)
    pack.add_argument('--patch',type=Path,required=True)
    restore=choices.add_parser('restore')
    for key in ('archive','patch','out'):restore.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='pack':
        study=args.study.resolve()
        base=study/'camera-batch';corrected=study/'corrected'
        manifest=corrected/'credits.json'
        rows=indexed(json.loads(manifest.read_bytes()))
        originals=indexed(json.loads((base/'credits.json').read_bytes()))
        report=json.loads((study/'policy/report.json').read_bytes())
        if rows.keys()!=originals.keys() or digest(manifest)!=report['manifest_sha256']:raise ValueError('Unexpected optimizer manifest')
        with zipfile.ZipFile(args.patch,'w',zipfile.ZIP_STORED) as out:
            out.write(manifest,'credits.json')
            out.write(corrected/'clock-audit.json','clock-audit.json')
            for name,entry in rows.items():
                patch=None
                if entry['title']==13:
                    with np.load(corrected/name,allow_pickle=False) as actual:
                        buff=io.BytesIO()
                        np.savez_compressed(buff,native_ticks_earned=actual['native_ticks_earned'],reward=actual['reward'])
                        patch=buff.getvalue()
                    out.writestr('th13-rewards/'+name,patch)
                rebuild((base/name).read_bytes(),entry,patch,originals[name])
            out.writestr('integrity.json',json.dumps(dict(raw_manifest_sha256=digest(base/'credits.json'),
                         corrected_manifest_sha256=digest(manifest),policy_sha256=report['policy_sha256']),indent=2)+'\n')
        print('Reconstructed and verified',len(rows),'optimizer shards;',args.patch.stat().st_size,'correction bytes',flush=True)
    else:
        args.out.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(args.archive) as raw,zipfile.ZipFile(args.patch) as delta:
            hashes=json.loads(delta.read('integrity.json'))
            manifests=[bundle.read('credits.json') for bundle in (raw,delta)]
            if [hashlib.sha256(x).hexdigest() for x in manifests]!=[hashes['raw_manifest_sha256'],hashes['corrected_manifest_sha256']]:
                raise ValueError('Camera archive and corrections do not match')
            originals,entries=[indexed(json.loads(v)) for v in manifests]
            if originals.keys()!=entries.keys():raise ValueError('Changed cohort membership')
            for name,row in entries.items():
                target=args.out/name
                if target.exists():
                    if digest(target)!=row['sha256']:raise ValueError(f'Different existing output {target}')
                    continue
                patch=delta.read('th13-rewards/'+name) if row['title']==13 else None
                result=rebuild(raw.read(name),row,patch,originals[name])
                with target.open('xb') as f:f.write(result)
            evidence=args.out/'credits.json'
            if evidence.exists() and evidence.read_bytes()!=manifests[1]:raise ValueError('Different output manifest exists')
            evidence.write_bytes(manifests[1])
        print('Restored',len(entries),'verified optimizer-ready shards',flush=True)


if __name__=='__main__':
    main()
