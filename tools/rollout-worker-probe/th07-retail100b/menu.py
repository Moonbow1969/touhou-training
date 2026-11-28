from pathlib import Path
import subprocess,numpy as np
from PIL import Image
base=Path('/var/tmp/moonbow-th07-retail-100b-preflight')
def cmd(name,s):subprocess.run(['docker','exec',name,'sh','-ec',s],check=True)
def key(name,k,delay=.5):cmd(name,'xdotool keydown '+k+'; sleep .12; xdotool keyup '+k+'; sleep '+str(delay))
def snap(name,f):cmd(name,'scrot -o /capture/'+f+'.png')
def title_score(f):
 with Image.open(f)as a: a=np.asarray(a.convert('L').crop((15,50,399,160)).resize((100,32)),'f').ravel()
 with Image.open(base/'tools/title-ref.png')as b:b=np.asarray(b.convert('L').crop((15,50,399,160)).resize((100,32)),'f').ravel()
 a-=a.mean();b-=b.mean();return (a@b)/max(1,np.linalg.norm(a)*np.linalg.norm(b))
