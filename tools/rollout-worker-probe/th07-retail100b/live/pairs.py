"""Run the fixed ordinary-start retail pairs after two sealed live qualification credits."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from boot import boot,root
from post import post
from verify_record import verify

study=root/'record'
tool=study/'tool'
plan=json.loads((study/'held-protocol.json').read_bytes())

def write(path,doc):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(doc,indent=2)+'\n')
    temp.replace(path)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run_credit(seed,arm):
    folder=study/f'{seed}-{arm}'
    try:
        if not folder.exists() or not any(folder.iterdir()):
            boot(dict(seed=seed,arm=arm,sha256=plan['models'][arm],role='held'))
        inflight=folder/'recovery-live.pid'
        if inflight.exists() and not (folder/'finished-game.json').exists():
            pid=int(inflight.read_text())
            command=Path(f'/proc/{pid}/cmdline')
            while command.exists() and not (folder/'finished-game.json').exists():
                args=command.read_bytes().split(b'\0')
                if not {b'play_record.py',str(seed).encode(),arm.encode()} <= {Path(a.decode()).name.encode() for a in args if a}:break
                time.sleep(1)
            if not (folder/'finished-game.json').exists():raise RuntimeError('In-flight original did not produce a terminal receipt')
        if not (folder/'finished-game.json').exists():
            with (folder/'execution.log').open('a') as log:
                subprocess.run([sys.executable,'-u',str(tool/'play_record.py'),str(seed),arm],
                               env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=log,
                               stderr=subprocess.STDOUT,check=True,timeout=2700)
        # Endings show the original event/credits and have their own navigation
        # and final ranking. Leave the original live for independent inspection.
        if (folder/'native-ending-state.txt').exists():
            print('ORIGINAL ENDING: preserve display, awaiting final ranking and saved original',folder,flush=True)
            while not (folder/'verified.json').is_file():time.sleep(15)
            return json.loads((folder/'verified.json').read_bytes())
        if not (folder/'ui-verified.json').exists():post(folder)
        return verify(folder,True)
    finally:
        if folder.exists() and (folder/'ready.json').exists():
            name=(folder/'actual-container').read_text().strip() if (folder/'actual-container').exists() else f'moonbow-th07-retail-record-{seed}'
            info=subprocess.run(['docker','inspect',name],capture_output=True)
            if info.returncode==0:
                mounts=json.loads(info.stdout)[0]['Mounts']
                if any(m['Destination']=='/capture' and Path(m['Source'])==folder.resolve() for m in mounts):
                    subprocess.run(['docker','stop','-t','1',name],capture_output=True)

def run_pair(seed):
    primary=seed not in plan['technical_reserves']
    i=seed-2170751 if primary else seed-2170771
    first_parent=(i%4 in (0,3)) if primary else (i%2==1)
    arms=['parent','sft-71281'] if first_parent else ['sft-71281','parent']
    result=dict(seed=seed,slot=i%2,order=arms,arms={})
    destination=study/f'pair-{seed}.json'
    if destination.exists():return json.loads(destination.read_bytes())
    for arm in arms:
        try:
            r=run_credit(seed,arm)
            result['arms'][arm]=dict(valid=r['passed'],ticks=r['stage1_native_ticks'],
                                     missing_windows=len(r['stale']),exact=r['exact_fraction'])
        except Exception as exc:
            result['arms'][arm]=dict(valid=False,error=repr(exc))
            print('TECHNICAL',seed,arm,repr(exc),flush=True)
    result['valid']=all(q['valid'] for q in result['arms'].values())
    write(destination,result)
    print('PAIR',seed,result['valid'],flush=True)
    return result

def main():
    reports=[]
    for identity in ('2170701-parent','2170702-sft-71281'):
        path=study/identity/'verified.json'
        q=json.loads(path.read_bytes())
        assert q['passed'] and not q['stale'] and q['terminal_visually_confirmed'] and q['terminal_continues']==0
        reports.append(dict(identity=identity,report_sha256=digest(path),exact=q['exact_fraction']))
    write(study/'qualification-release.json',dict(held_protocol_sha256=digest(study/'held-protocol.json'),qualifications=reports))
    pairs=[]
    with ThreadPoolExecutor(2) as pool:
        for j in range(0,len(plan['primary_seeds']),2):
            pairs.extend(pool.map(run_pair,plan['primary_seeds'][j:j+2]))
    reserves=iter(plan['technical_reserves'])
    def valid():return [p for p in pairs if p['valid']]
    def fill(n):
        while len(valid())<n:
            if max(sum(not p['arms'][a]['valid'] for p in pairs)for a in plan['models'])>2:raise RuntimeError('Technical failure limit')
            spare=next(reserves,None)
            if spare is None:raise RuntimeError('Preregistered technical reserve exhausted')
            pairs.append(run_pair(spare))
    fill(6)
    def exposure():return {a:sum(p['arms'][a]['ticks']for p in valid())for a in plan['models']}
    for seed in plan['extension_seeds']:
        if min(exposure().values())>=30000:break
        n=len(valid())+1;pairs.append(run_pair(seed));fill(n)
    write(study/'completed-pairs.json',dict(pairs=pairs,valid_pairs=len(valid()),exposure=exposure()))

if __name__=='__main__':
    with (study/'pair-dispatch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
