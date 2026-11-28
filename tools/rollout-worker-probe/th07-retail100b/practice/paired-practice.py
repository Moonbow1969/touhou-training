"""Run one private lane from a precommitted paired Normal TH07 original practice registration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser()
    for key in ('study','driver','game-dir','thprac','probe','actor','parent','candidate'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--image',required=True)
    p.add_argument('--wsl')
    p.add_argument('--lane',choices=('private-practice-a','private-practice-b'),required=True)
    args=p.parse_args();d=args.study.resolve()
    path=d/'paired-registration.json';reg=json.loads(path.read_bytes())
    binding_path=d/'released-binding.json';binding=json.loads(binding_path.read_bytes())
    if not binding.get('released') or binding.get('paired_registration_sha256')!=sha(path):raise RuntimeError('Frozen held release absent')
    policies={'parent':args.parent,'practice-ppo':args.candidate}
    for arm,actor in policies.items():
        if sha(actor)!=binding['policies'][arm]:raise ValueError(f'Changed model {arm}')
        if sha(d/f'arm-{arm}.json')!=binding['registered_arm_sha256'][arm]:raise ValueError('Invalid per-arm registration')
    for f,digest in reg['driver_sha256'].items():
        if sha(args.driver/f)!=digest:raise ValueError(f'Changed qualified camera/menu source {f}')
    if sha(args.actor)!=reg['native_worker_sha256']:raise ValueError('Changed original native-mode worker')
    (d/'logs').mkdir(exist_ok=True)
    (d/'raw').mkdir(exist_ok=True)
    os.environ.update(TH07_PHYSICAL_LANE=args.lane,TH07_PRACTICE_CPUSET=reg['cpus'][args.lane])
    dispatch=d/('dispatch-'+args.lane+'.ndjson')
    for row in sorted((v for v in reg['rows'] if v['lane']==args.lane),key=lambda v:v['lane_pair']):
        for pos,arm in enumerate(row['order']):
            seed=row['seed'];tag=f'{seed}-{arm}';case=d/'raw'/tag
            if (case/'qualification.json').is_file() or (case/'attempt-failed.json').is_file():continue
            if case.exists():raise RuntimeError('Prior unfinished original must be independently reconciled: '+tag)
            if shutil.disk_usage(d).free<3*1024**3:raise RuntimeError('Physical volume less than 3 GiB free')
            if sum(f.stat().st_size for f in (d/'raw').rglob('*') if f.is_file())>1500*1024**2:raise RuntimeError('Study exceeded 1.5 GiB raw limit')
            launched=time.time()
            event=dict(seed=seed,arm=arm,host=row['host'],lane=row['lane'],position=pos+1)
            with dispatch.open('a') as fp:fp.write(json.dumps(dict(event,event='launch',time=launched))+'\n')
            cmd=[sys.executable,str(args.driver/'practice_reset.py'),'--game-dir',str(args.game_dir),'--thprac',str(args.thprac),'--probe',str(args.probe),'--actor',str(args.actor),'--out',str(case),'--image',args.image,'--scenario',row['scenario']]
            if args.wsl:cmd+=['--wsl',args.wsl]
            cmd+=['--worker',sys.executable,str(args.driver/'learned-worker.py'),str(policies[arm]),str(d/f'arm-{arm}.json'),str(seed)]
            log=d/'logs'/(tag+'.log')
            with log.open('xb') as stream:res=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
            if res.returncode:
                failed=dict(event,start_epoch=launched,end_epoch=time.time(),returncode=res.returncode,log_sha256=sha(log),evidence={f.name:sha(f) for f in case.iterdir() if f.is_file()})
                (case/'attempt-failed.json').write_text(json.dumps(failed,indent=2)+'\n')
            with dispatch.open('a') as fp:fp.write(json.dumps(dict(event,event='sealed' if res.returncode==0 else 'failed',time=time.time()))+'\n')
    (d/('complete-'+args.lane+'.json')).write_text(json.dumps(dict(lane=args.lane,registration_sha256=sha(path),release_sha256=sha(binding_path),end_epoch=time.time()))+'\n')
if __name__=='__main__':main()
