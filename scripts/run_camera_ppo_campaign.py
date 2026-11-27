"""Advance a synchronous multi-title PPO campaign when complete frozen-actor credits arrive."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.checkpoint_storage import digest
from scripts.score_original_camera_run import records


def atomic(path,data):
    temp=path.with_suffix('.partial')
    temp.write_text(json.dumps(data,indent=2)+'\n',encoding='utf8');temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--poll-seconds',type=int,default=15)
    a=p.parse_args();root=a.root.resolve();campaign=root/'campaign';campaign.mkdir(exist_ok=True)
    basis=json.loads((root/'camera-protocol.json').read_bytes());plans={}
    for g in (3,4,5):
        base={t:800000+10000*g+100*t for t in (10,12,13)}
        plans[g]={**basis,'generation':g,'name':f'Camera PPO generation {g} on 72 new original credits',
                  'train_seeds':{t:list(range(v+1,v+25)) for t,v in base.items()},
                  'eval_seeds':{t:list(range(v+51,v+63)) for t,v in base.items()} if g==5 else {},
                  'reference_arm':'camera-ppo2','candidate':f'camera-ppo{g}',
                  'registered_parent':f'camera-ppo{g-1}','registered_train_credits_per_title':24,
                  'selection':'Fixed training budget and same reward. No checkpoint selection against held outcomes. Final paired live test compares generation 5 with generation 2.'}
        folder=campaign/f'g{g:02}';folder.mkdir(exist_ok=True)
        atomic(folder/'camera-protocol.json',plans[g])
    atomic(campaign/'registration.json',plans)
    for g,plan in plans.items():
        folder=campaign/f'g{g:02}';parent=root/plan['registered_parent'];dest=root/f'camera-ppo{g}'
        while not (parent/'report.json').is_file():time.sleep(a.poll_seconds)
        fingerprint=digest(parent/'policy.ts')
        if (dest/'report.json').is_file():
            if json.loads((dest/'report.json').read_bytes())['native_behavior_sha256']!=fingerprint:
                raise ValueError('Published output has a different parent')
            continue
        wanted={(int(t),seed) for t,seeds in plan['train_seeds'].items() for seed in seeds}
        previous=None
        while True:
            found=set()
            for t,f in records(folder,True,True):
                try:r=json.loads(f.read_bytes())
                except (ValueError,PermissionError):continue
                key=(t,r.get('seed'))
                if key not in wanted:continue
                if r.get('policy_sha256')!=fingerprint:raise ValueError(f'Wrong policy served for {key}')
                if key in found:raise ValueError(f'Duplicate original credit {key}')
                path=f.parent/r.get('behavior',r.get('trajectory',r.get('file','missing')))
                if path.is_file():found.add(key)
            if found!=previous:
                progress={k:sum(t==k for t,s in found) for k in (10,12,13)}
                atomic(campaign/'status.json',dict(generation=g,phase='collecting',counts=progress,parent=fingerprint))
                print('generation',g,'credits',progress,flush=True);previous=found
            if found==wanted:break
            time.sleep(a.poll_seconds)
        if shutil.disk_usage(campaign).free<5*(1024**3):raise RuntimeError('Insufficient room to pack original frames')
        pack=[sys.executable,'-m','scripts.pack_camera_ppo_rollouts','--root',folder,'--round2','--behavior',parent/'policy.ts',
              '--protocol',folder/'camera-protocol.json','--out',folder/'camera-batch']
        subprocess.run(list(map(str,pack)),check=True)
        fit=[sys.executable,'-m','scripts.train_full_resolution_camera_ppo','--manifest',folder/'camera-batch/credits.json',
             '--protocol',folder/'camera-protocol.json','--parent-actor-critic',parent/'camera-ppo.pt',
             '--behavior',parent/'policy.ts','--out',dest]
        subprocess.run(list(map(str,fit)),check=True)
        report=json.loads((dest/'report.json').read_bytes())
        atomic(campaign/'status.json',dict(generation=g,phase='trained',steps=report['steps'],checkpoint=report['policy_sha256']))
        print('trained',g,report['steps'],report['policy_sha256'],flush=True)


if __name__=='__main__':main()
