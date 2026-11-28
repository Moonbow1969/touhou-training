from concurrent.futures import ThreadPoolExecutor
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
def lane(q):
    seed=q['seed'];folder=study/f"{seed}-{q['arm']}"
    boot(q)
    name=(folder/'actual-container').read_text().strip()
    engine=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
    pid=engine['State']['Pid']
    cgroup=Path('/sys/fs/cgroup')/Path(f'/proc/{pid}/cgroup').read_text().strip().split(':',2)[2].lstrip('/')
    assert (cgroup/'cpu.max').read_text().split()[0]=='max'
    assert engine['HostConfig']['CpusetCpus']==','.join(map(str,q['physical_cpu_ids']))
    def snapshot():
        return dict(ns=time.monotonic_ns(),cpu=(cgroup/'cpu.stat').read_text(),cpu_pressure=(cgroup/'cpu.pressure').read_text(),
                    memory=(cgroup/'memory.events').read_text(),swap=(cgroup/'memory.swap.current').read_text())
    with (folder/'execution.log').open('w')as log,(folder/'cgroup.jsonl').open('w',buffering=1) as trace:
        worker=subprocess.Popen(['taskset','-c',engine['HostConfig']['CpusetCpus'],sys.executable,'-u',str(study/'tool/play_record.py'),str(seed)],
                                stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
        deadline=time.monotonic()+2700
        while worker.poll() is None:
            print(json.dumps(snapshot()),file=trace)
            if time.monotonic()>deadline:raise TimeoutError('Original engineering credit exceeded 45 minutes')
            time.sleep(.25)
        assert worker.returncode==0
    if (folder/'native-ending-state.txt').exists():
        raise RuntimeError('Preserve original Ending for independent navigation')
    post(folder)
    result=verify(folder,True)
    print('QUALIFICATION',seed,result['passed'],flush=True)
    info=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
    assert any(x['Destination']=='/capture' and Path(x['Source'])==folder.resolve() for x in info['Mounts'])
    subprocess.run(['docker','stop','-t','1',name],check=True)
    return dict(seed=seed,arm=q['arm'],qualified=result['passed'],cgroup_cpu_max=(engine['HostConfig']['NanoCpus']),
                pinned_cpu_ids=q['physical_cpu_ids'],original_cuda=False,strict_fraction=result['exact_fraction'],
                skipped=result['skipped_decisions'],max_native_read_ms=result['max_synchronous_observer_ms'])

if __name__=='__main__':
    assert (root.parent/'action-preserving-native/record/completed-pairs.json').exists()
    plan=json.loads((study/'protocol.json').read_bytes())
    rows=[]
    def attempted(q):
        try:
            return lane(q)
        except Exception as exc:
            import traceback
            folder=study/f"{q['seed']}-{q['arm']}"
            (folder/'engineering-error.txt').write_text(traceback.format_exc())
            return dict(seed=q['seed'],qualified=False,error=repr(exc))
    for seeds in plan['waves']:
        with ThreadPoolExecutor(2) as pool:
            rows.extend(pool.map(attempted,[q for q in plan['lanes'] if q['seed'] in seeds]))

    (study/'engineering-results.json').write_text(json.dumps(rows,indent=2)+'\n')
