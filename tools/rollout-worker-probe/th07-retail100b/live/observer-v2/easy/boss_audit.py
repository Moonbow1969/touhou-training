from pathlib import Path
from PIL import Image,ImageOps,ImageDraw
import numpy as np,gzip,struct,json,csv,hashlib
from PIL import ImageFilter
r=Path('/var/tmp/moonbow-th07-retail-100b-preflight');base=r/'easy-qualification';back=r/'preflight/green-v2-s1/full-color.bgra.gz'
def contrast(im):
 roi=im.convert('L').crop((36,30,119,45));a=np.asarray(roi,dtype=float)-np.asarray(roi.filter(ImageFilter.GaussianBlur(1.3)),dtype=float)
 return a.ravel()/max(1,np.linalg.norm(a))
refs={}
with gzip.open(back,'rb') as f:
 while pkt:=f.read(4):
  tick=struct.unpack('<I',pkt)[0];data=f.read(640*480*4)
  if tick in [1200,2400,3600,4800,7200,8400,9600,10800]:refs[tick]=contrast(Image.frombytes('RGBA',(640,480),data,'raw','BGRA').convert('RGB'))
# Freeze reference from three separate spell/wave backgrounds; the fourth positive is reserved to audit transfer.
m=np.median(np.array([refs[t] for t in (7200,8400,9600)]),axis=0);m/=np.linalg.norm(m)
cal={t:float(s@m) for t,s in refs.items()}
print('reference',cal)
for p in sorted((base/'record').glob('217*-parent')):
 frames=sorted((p/'boss-frames').glob('*.jpg'));scores=[]
 for f in frames:
  try:scores.append([f.name,float(contrast(Image.open(f))@m)])
  except OSError:continue
 ix=[i for i in range(len(scores)-1) if min(scores[i][1],scores[i+1][1])>.70]
 print(p.name,'first-reference-match', scores[ix[0]:ix[0]+2] if ix else None, 'max',max([s for f,s in scores],default=0.))
 (base/(p.name+'-boss-scores.json')).write_text(json.dumps(dict(reference=cal,screen_name_scores=scores),indent=2)+'\n')
