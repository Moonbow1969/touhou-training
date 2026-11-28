"""Capture only registered original human source bytes, on two fixed Momentum lanes."""
from pathlib import Path
import argparse,contextlib,fcntl,hashlib,json,shutil,subprocess,time
import numpy as np
from PIL import Image
base=Path('/var/tmp/moonbow-th07-trial-261127')
root=base/'human-normal-shots'
plan=json.loads((root/'registration.json').read_bytes())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def original(slug,stage):
 rec=next(v for v in plan['records']if v['slug']==slug)
 out=root/slug/f's{stage}'
 if (out/'capture.json').is_file():return
 out.mkdir(parents=True,exist_ok=False)
 source=root/'sources'/rec['source_file']
 assert digest(source)==rec['sha256']
 shutil.copyfile(source,out/'th7_01.rpy')
 name=f'moonbow-th07-humans-{slug}-s{stage}'
 args=['docker','run','-d','--rm','--name',name,'--network','none','--cpus','2','--memory','1750m','--pids-limit','300','--cap-add','SYS_PTRACE','--tmpfs','/tmp:rw,exec,size=1300m',
       '-v',f'{base}/input/youmu:/input:ro','-v',f'{base}/human/launch-public-rpy.sh:/start.sh:ro','-v',f'{out}:/capture','-v',f'{root}/tools:/tool:ro','-e','DISPLAY=:73','--entrypoint','sh','toho-trial:endeavor-native-v6','/start.sh']
 def cmd(s):subprocess.run(['docker','exec',name,'sh','-ec',s],check=True)
 def key(k):cmd('xdotool keydown '+k+'; sleep .1; xdotool keyup '+k+'; sleep .72')
 def snap(n):cmd('scrot -o /capture/'+n+'.png')
 def arr(im,box):return np.asarray(Image.open(im).convert('L').crop(box).resize((96,48)),dtype='f').ravel()
 def similar(p,q,box):
  a,b=arr(p,box),arr(q,box);a-=a.mean();b-=b.mean()
  return a@b/max(1,np.linalg.norm(a)*np.linalg.norm(b))
 subprocess.run(args,check=True)
 try:
  for i in range(100):
   if (out/'window').is_file():break
   time.sleep(.5)
  else:raise ValueError('No original window')
  for i in range(42):
   snap('title')
   if similar(out/'title.png',base/'virtual/identity.png',(25,45,410,172))>.85:break
   time.sleep(.35)
  else:raise ValueError('Mismatch original title')
  time.sleep(1.7);key('Down');key('z');key('z');snap('stage-selector')
  if stage==2:key('Down')
  snap('selected-stage');key('z');snap('normal-playback-mode')
  subprocess.run(['docker','exec','-d',name,'sh','-c','/tool/sensor >/capture/sensor.log 2>&1; echo $? >/capture/sensor.exit'],check=True)
  key('z');snap('opening')
  for i in range(425):
   if (out/'sensor.exit').is_file():break
   time.sleep(1)
  else:raise TimeoutError('Did not leave selected original stage')
  snap('after-stage')
  assert (out/'sensor.exit').read_text().strip()=='0'
  metadata=dict(role=rec['role'],author=rec['archive_author'],slug=slug,shot=rec['shot'],rank='Normal',stage=stage,
     source_sha256=rec['sha256'],game_build=(out/'build.sha256').read_text(),registration_sha256=digest(root/'registration.json'))
  (out/'capture.json').write_text(json.dumps(metadata,indent=2)+'\n')
 finally:
  subprocess.run(['docker','stop','-t','2',name],check=False)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('lane',type=int,choices=(0,1));a=parser.parse_args()
 queue=[('pykrete-reimub',1),('pykrete-reimub',2),('pykrete-marisaa',1)]if a.lane==0 else [('xiongataosenai-sakuyaa',1),('xiongataosenai-sakuyaa',2),('pykrete-marisaa',2)]
 # A lane waits until its prior frozen-author Stage 2 capture physically closes.
 oldrole=['train','held'][a.lane];old=base/'human-02-expansion-v3'/oldrole
 for i in range(450):
  if (old/'after-stage.png').is_file() and (old/'sensor.exit').is_file():break
  time.sleep(1)
 else:raise TimeoutError('Prior original stage has not sealed')
 oldname='moonbow-th07-human-stage2-261128-v3-'+oldrole
 subprocess.run(['docker','stop','-t','2',oldname],check=False)
 for slug,stage in queue:original(slug,stage)
