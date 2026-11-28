"""Acquire one registered original retail stage in a fresh isolated Wine/Xvfb lane."""
from pathlib import Path
import argparse,hashlib,subprocess,time,shutil,json
from menu import base,key,snap,cmd,title_score
def run(source,stage,subdir):
    study=base
    out=study/subdir
    out.mkdir(parents=True,exist_ok=False)
    src=study/'sources'/source
    shutil.copyfile(src,out/'th7_01.rpy')
    name='moonbow-th07-retail-'+subdir.replace('/','-')
    command=['docker','run','-d','--name',name,'--rm','--network','none','--cpus','2','--memory','1700m','--pids-limit','300','--cap-add','SYS_PTRACE','--tmpfs','/tmp:rw,exec,size=1250m','-v',f'{study}/input:/input:ro','-v',f'{study}/tools:/tool:ro','-v',f'{out}:/capture','-e','DISPLAY=:73','--entrypoint','sh','toho-trial:endeavor-native-v6','/tool/start.sh']
    subprocess.run(command,check=True)
    try:
        for i in range(160):
            if (out/'window').exists():break
            time.sleep(.35)
        else:raise ValueError('Window failed to open')
        cmd(name,'xdotool windowfocus "$(cat /capture/window)"')
        for i in range(48):
            snap(name,'title-ready')
            if title_score(out/'title-ready.png')>.87:break
            time.sleep(.3)
        else:raise ValueError('Original retail title not reached')
        time.sleep(1.5)
        for k in ('Down','Down'):key(name,k)
        snap(name,'replay-selected')
        key(name,'z');snap(name,'source-list')
        key(name,'z');snap(name,'stage-selector')
        if stage==2:key(name,'Down')
        snap(name,'selected-stage')
        key(name,'z');snap(name,'playback-mode')
        subprocess.run(['docker','exec','-d','-e',f'TH07_STAGE={stage}',name,'sh','-c','/tool/sensor-v2 >/capture/sensor.log 2>&1; echo $? >/capture/sensor.exit'],check=True)
        key(name,'z',1.7);snap(name,'opening')
        for i in range(602):
            if (out/'sensor.exit').is_file():break
            time.sleep(1)
        else:raise TimeoutError('Native replay stage has not exited')
        snap(name,'after-stage')
        assert (out/'sensor.exit').read_text().strip()=='0'
        (out/'capture.json').write_text(json.dumps(dict(stage=stage,source=source,sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
            original=(out/'build.sha256').read_text(),exit='Original stage change or replay-manager reset to zero observed after >1000 ticks',actual_lane=name),indent=2)+'\n')
    finally:
        subprocess.run(['docker','stop','-t','1',name],check=False)
if __name__=='__main__':
    import sys
    run(sys.argv[1],int(sys.argv[2]),sys.argv[3])
