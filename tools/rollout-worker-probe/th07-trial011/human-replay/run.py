from pathlib import Path
import subprocess,time,sys,shlex,hashlib
import numpy as np
from PIL import Image
base=Path('/var/tmp/moonbow-th07-trial-261127')
area=base/'human-02-expansion-v3'
lane=sys.argv[1]
assert lane in ('train','held')
folder=area/lane
folder.mkdir(exist_ok=False)
old=base/'human'/('capture-'+lane)
subprocess.run(['cp',str(old/'th7_01.rpy'),str(folder/'th7_01.rpy')],check=True)
sha=hashlib.sha256((folder/'th7_01.rpy').read_bytes()).hexdigest()
assert sha=={'train':'219b2a89be16720447df65fd035618f5926af53d7e8337d5f61e07842ba3fb94','held':'9da50f8af1d33d3db5cfdb2f80c9d16143d9538ec0be3ceab33f223c31a5a64f'}[lane]
name='moonbow-th07-human-stage2-261128-v3-'+lane
subprocess.run(['docker','run','-d','--name',name,'--network','none','--cpus','2','--memory','1750m','--pids-limit','300','--cap-add','SYS_PTRACE','--tmpfs','/tmp:rw,exec,size=1300m',
 '-v',f'{base}/input/youmu:/input:ro','-v',f'{base}/human/launch-public-rpy.sh:/start.sh:ro','-v',f'{folder}:/capture','-v',f'{area}:/tool:ro','-e','DISPLAY=:73','--entrypoint','sh','toho-trial:endeavor-native-v6','/start.sh'],check=True)
def cmd(s): subprocess.run(['docker','exec',name,'sh','-ec',s],check=True)
def key(k):cmd('xdotool keydown '+k+'; sleep .1; xdotool keyup '+k+'; sleep .72')
def snap(n):cmd('scrot -o /capture/'+n+'.png')
def arr(p,box):return np.asarray(Image.open(p).convert('L').crop(box).resize((96,48)),dtype='f').ravel()
def match(p,ref,box):
 a=arr(p,box);b=arr(ref,box);a-=a.mean();b-=b.mean();return a@b/max(1,np.linalg.norm(a)*np.linalg.norm(b))
for i in range(100):
 if (folder/'window').exists():break
 time.sleep(.5)
else:raise RuntimeError('No native window')
for i in range(42):
 snap('title')
 if match(folder/'title.png',base/'virtual/identity.png',(25,45,410,172))>.85:break
 time.sleep(.35)
else:raise RuntimeError('No original title')
time.sleep(1.7);key('Down');key('z');key('z');snap('stage-selector')
assert match(folder/'stage-selector.png',old/'matched-stage-selector.png',(8,125,365,363))>.94,'Source/stage original menu mismatch'
key('Down');snap('selected-stage2');key('z');snap('normal-playback-mode')
for i in range(240):
 if (area/'sensor').exists():break
 time.sleep(.5)
else:raise RuntimeError('Sensor unavailable')
subprocess.run(['docker','exec','-d',name,'sh','-c','/tool/sensor >/capture/sensor.log 2>&1; echo $? >/capture/sensor.exit'],check=True)
key('z');snap('opening')
for i in range(425):
 if (folder/'sensor.exit').exists():break
 time.sleep(1)
else:raise RuntimeError('Sensor timeout')
snap('after-stage')
print('exit', (folder/'sensor.exit').read_text(),flush=True)



