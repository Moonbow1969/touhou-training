import sys,importlib.util
from pathlib import Path
import numpy as np
base=Path('/var/tmp/moonbow-th07-trial-261127');root=base/'human-02-expansion-v3'
sys.path.insert(0,str(base/'online'))
from inspect_rpy import decompress
import struct
def keys(p):
 b=bytearray(p.read_bytes())
 assert bytes(b[:6])==bytes.fromhex('543752501110')
 b[16:]=bytes((v-b[13]-7*i)%256 for i,v in enumerate(b[16:]))
 size,ds,*st=struct.unpack_from('<9I',b,20)
 trailing=struct.unpack_from('<7I',b,56)
 d=decompress(bytes(b[84:]),ds)
 stage=int(p.parent.name[1:]) if p.parent.name in ('s1','s2') else 2
 start=st[stage-1]
 end=min(a for a in [*st,*trailing,len(d)+84] if a>start)
 return np.frombuffer(d[start-84+40:end-84],'<u2').reshape(-1,2)[:,0].copy()
def xy(p):
 a=[(int(k,16),float(s))for k,s in [l.split() for l in (p/'candidates.tsv').read_text().splitlines()]]
 raw=np.fromfile(p/'native.bin','<u4');assert len(raw)%(2*len(a))==0;n=len(raw)//(2*len(a));w=raw[:n*len(a)*2].reshape(n,len(a),2)
 f=np.loadtxt(p/'clock.csv',dtype=np.int64,delimiter=',');assert len(f)==len(w)
 src=keys(p/'th7_01.rpy');assert (src[f[:,0]]==f[:,1]).all() and (src[f[:,0]-1]==f[:,2]).all()
 def pos(addr):
  j=next(j for j,(k,s)in enumerate(a)if k==addr)
  return w[:,j].view('<f4') if a[j][1]>0 else w[:,j].view('<i4').astype('f4')/-a[j][1]
 return f,src,a,pos
def support(t,src,co):
 df=np.diff(co,axis=0); t0=t[:-1];t1=t[1:];valid=t1-t0==4
 out={}
 for offset in range(-3,3):
  at=t0[:,None]+np.arange(1,5)+offset
  if at.min()<0 or at.max()>=len(src):continue
  bits=src[at]
  dx=((bits&128>0).astype(int)-(bits&64>0).astype(int));dy=((bits&32>0).astype(int)-(bits&16>0).astype(int))
  flat=[]
  for axis,val in enumerate([dx,dy]):
   nom=val.sum(1)
   observed=df[:,axis]
   # Includes any valid four-tick window where both signed axis displacements are observable.
   usable=valid&(abs(observed)>.02)&(abs(observed)<30)&(nom!=0)
   flat.extend(np.sign(nom[usable])==np.sign(observed[usable]))
  out[offset]=(len(flat),round(float(np.mean(flat)),5))if flat else (0,None)
 return out
if __name__=='__main__':
 p=root/'train';f,src,a,pos=xy(p);print('rows',len(f),'latest',f[-1,0])
 for addr,sc in a:
  l=pos(addr)
  print(hex(addr),sc,l[0],l[-1],np.round(np.ptp(l,axis=0),1),support(f[:,0],src,l).get(0))



