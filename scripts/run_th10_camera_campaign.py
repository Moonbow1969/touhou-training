"""Collect prespecified native TH10 whole-credit batches for direct camera PPO generations 3–5."""
import argparse
from pathlib import Path
import shutil
from types import SimpleNamespace
import zipfile

import torch

from scripts.run_th10_camera_cohort import checkpoint,collect,verify,save
from scripts.run_th10_camera_pair import run as paired
from scripts.package_th10_credit_logs import package


def campaign(args):
    root=args.root.resolve();workers=args.workers.resolve()
    jobs=SimpleNamespace(host='fleet',workers=workers)
    torch.set_num_threads(2)
    (root/'campaign').mkdir(exist_ok=True)
    save(root/'campaign/th10-preregistration.json',dict(
        train={f'g{g:02d}':list(range(801001+10000*g,801025+10000*g)) for g in (3,4,5)},
        held=list(range(851051,851063)),held_arms=['camera-ppo2/policy.ts','camera-ppo5/policy.ts'],
        host='Odd controller seed Accelerator; even Momentum; same host within held pair.',
        source='Original official TH10 Normal trial, stock lives, no assistance, no Continue.'))
    for generation in (3,4,5):
        base=800000+10000*generation+100*10
        seeds=list(range(base+1,base+25))
        directory=root/f'campaign/g{generation:02d}/native10'
        directory.mkdir(parents=True,exist_ok=True)
        policy=root/f'camera-ppo{generation-1}/policy.ts'
        save(directory/'training-protocol.json',dict(generation=generation,title=10,seeds=seeds,
            parent=f'camera-ppo{generation-1}/policy.ts',host_assignment='Odd Accelerator; even Momentum',
            global_concurrency=args.concurrency,control_hold_ms=120,resources='Original Normal full credit, stock lives, all actions, no Continue or assistance',
            archives='ZIP-deflated frame/timestamp and behavior NPZ, gzip original native telemetry, no game files; immutable per-credit SHA checked after extraction.'))
        model=checkpoint(policy)
        free=shutil.disk_usage(root).free
        if free<8*1024**3:raise IOError('The staging volume has less than eight GiB free.')
        collect(jobs,f'camera-ppo{generation-1}-train-g{generation:02d}',policy,seeds,directory,args.concurrency,True)
        receipts,audit=verify(directory,policy,model,seeds,'train',write=False,host='fleet')
        sizes={}
        for key in ('behavior','pixels'):
            total=0;raw=0
            for r in receipts:
                path=Path(r[key]);total+=path.stat().st_size
                with zipfile.ZipFile(path) as z:
                    if z.testzip() is not None or any(f.compress_type!=zipfile.ZIP_DEFLATED for f in z.infolist()):
                        raise ValueError(f'Unverified lossless compressed arrays at {path}')
                    raw+=sum(f.file_size for f in z.infolist())
            sizes[key]=dict(compressed_bytes=total,uncompressed_npy_bytes=raw)
        save(directory/'on-policy-validation.json',audit)
        save(directory/'archive.json',dict(credits=len(receipts),decisions=sum(r['steps'] for r in receipts),
             local_free_bytes_before=free,local_free_bytes_after=shutil.disk_usage(root).free,arrays=sizes))
        save(directory/'ppo-receipts.json',receipts)
        package(directory.parent/'th10-native-logs.zip',[directory])
    seeds=list(range(851051,851063))
    paired(SimpleNamespace(workers=workers,out=root/'campaign/g05/native10',reference=root/'camera-ppo2/policy.ts',
        candidate=root/'camera-ppo5/policy.ts',reference_name='held-ppo2',candidate_name='held-ppo5',
        seeds=seeds,concurrency=args.concurrency,host='fleet',candidate_history=False))
    out=root/'campaign/g05/native10'
    package(out.parent/'th10-held-native-logs.zip',[out/'held-ppo2',out/'held-ppo5'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=Path,required=True)
    p.add_argument('--concurrency',type=int,default=6)
    campaign(p.parse_args())
