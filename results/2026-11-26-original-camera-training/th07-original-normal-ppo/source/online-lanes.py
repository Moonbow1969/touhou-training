"""Serial TH07 original credits per rank-fixed physical lane; publish exact raw records on the host."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

base=Path('/var/tmp/moonbow-th07-trial-261127')
online=base/'online'
plan=json.loads((online/'protocol.json').read_text())

def wait(path,timeout=840):
    deadline=time.monotonic()+timeout
    while not path.exists():
        if time.monotonic()>deadline:raise TimeoutError(str(path))
        time.sleep(.5)

def stop(name):
    state=subprocess.run(['docker','inspect','-f','{{.State.Running}}',name],capture_output=True,text=True)
    if state.returncode==0 and state.stdout.strip()=='true':
        subprocess.run(['docker','stop','-t','1',name],check=True,stdout=subprocess.DEVNULL)

def run(seed,arm,role,lane,retry=0):
    model=base/'policy/policy.ts' if arm=='six-ppo' else online/'policy/policy.ts'
    if arm=='th07-ppo':wait(online/'policy/report.json')
    sha=hashlib.sha256(model.read_bytes()).hexdigest()
    expected=plan['parent_policy_sha256'] if arm=='six-ppo' else json.loads((online/'policy/report.json').read_text())['policy_sha256']
    assert expected==sha
    name=f'moonbow-th07-208-{lane}-{role}-{seed}-{arm}'
    if retry:name+=f'-r{retry}'
    folder=online/(f'train-{seed}' if role=='train' else f'held-{arm}-{seed}')
    folder.mkdir(exist_ok=True)
    if (folder/'receipt.pending-visual.json').is_file():return
    command=['docker','run','-d','--name',name,'--network','none','--cpus','2','--memory','1600m','--pids-limit','300',
        '-v',f'{base}/input/youmu:/input:ro','-v',f'{base}/input/launch.sh:/start.sh:ro','-v',f'{folder}:/capture',
        '-e','DISPLAY=:73','--entrypoint','sh','toho-trial:endeavor-native-v6','/start.sh']
    subprocess.run(command,check=True)
    env=dict(os.environ,TH07_OUT=str(folder),TH07_SEED=str(seed),TH07_CONTAINER=name,TH07_SHA=sha,TH07_MODEL=str(model))
    with (folder/'runner.log').open('a') as log:
        result=subprocess.run(['python3','-u',str(online/'credit.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{name} needs inspection: {folder}/runner.log')
    stop(name)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('lane',type=int,choices=(0,1));a=p.parse_args();lane=a.lane
    first=plan['train'][lane]
    wait(online/f'train-{first}/receipt.pending-visual.json');stop(f'moonbow-th07-208-{first}')
    run(plan['train'][lane+2],'six-ppo','train',lane)
    seed=plan['held'][lane]
    first=plan['held_first'][lane]
    for arm in [first,'th07-ppo' if first=='six-ppo' else 'six-ppo']:
        run(seed,arm,'held',lane)
