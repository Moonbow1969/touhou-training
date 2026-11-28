from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json, subprocess, sys, time
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tools'))
from menu import key,snap,title_score,cmd

def boot(plan):
    seed,arm=plan['seed'],plan['arm']
    out=root/'record'/f'{seed}-{arm}'
    out.mkdir(exist_ok=True)
    if any(out.iterdir()):raise FileExistsError(out)
    name=f'moonbow-th07-retail-record-{seed}-{arm}'
    subprocess.run(['docker','run','-d','--rm','--name',name,'--network','none','--cpuset-cpus',','.join(map(str,plan['physical_cpu_ids'])),'--memory','1700m',
                    '--pids-limit','300','--cap-add','SYS_PTRACE','--tmpfs','/tmp:rw,exec,size=1250m',
                    '-v',f'{root}/input:/input:ro','-v',f'{root}/record/tool:/tool:ro','-v',f'{out}:/capture',
                    '-e','DISPLAY=:73','--entrypoint','sh','toho-trial:endeavor-native-v6','/tool/start.sh'],check=True)
    (out/'actual-container').write_text(name+'\n')
    for _ in range(180):
        if (out/'window').exists():break
        time.sleep(.25)
    else:raise RuntimeError('no original window')
    cmd(name,'xdotool windowfocus "$(cat /capture/window)"')
    for _ in range(80):
        snap(name,'title')
        if title_score(out/'title.png')>.87:break
        time.sleep(.3)
    else:raise RuntimeError('title not qualified')
    time.sleep(1.3)
    key(name,'z');snap(name,'normal-before')
    key(name,'Up');snap(name,'difficulty')
    for scene in ('character','shot'):
        key(name,'z');snap(name,scene)
    (out/'ready.json').write_text(json.dumps(plan)+'\n')
    print(seed,'ready in original Game Start shot menu',flush=True)

if __name__=='__main__':
    plan=json.loads((root/'record/protocol.json').read_bytes())
    with ThreadPoolExecutor(2)as pool:list(pool.map(boot,plan['lanes']))
