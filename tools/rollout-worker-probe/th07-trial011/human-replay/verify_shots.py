import gzip,hashlib,json,struct,sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from coord_shots import root,keys,xy,support
cv2.setNumThreads(1)
SHA={'train':'219b2a89be16720447df65fd035618f5926af53d7e8337d5f61e07842ba3fb94','held':'9da50f8af1d33d3db5cfdb2f80c9d16143d9538ec0be3ceab33f223c31a5a64f'}
def sha(p):
 with p.open('rb')as z:return hashlib.file_digest(z,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def color(p):
 with gzip.open(p/'full-color.bgra.gz','rb') as z:
  while True:
   tick=z.read(4)
   if not tick:return
   assert len(tick)==4
   rgb=np.frombuffer(z.read(640*480*4),'u1').reshape(480,640,4)[:,:,:3]
   yield struct.unpack('<I',tick)[0],rgb
def verified(p,meta):
 f,src,candidates,pos=xy(p);t=f[:,0];co=pos(0x1280e58).copy();curr=f[:,1];prev=f[:,2]
 assert sha(p/'th7_01.rpy')==meta['source_sha256']
 assert (p/'build.sha256').read_text().split()[0]=='2ca6b41ce2b07fb2b8a119f9683fc497dd724edcbc130b5076f8d86323b9091f'
 assert (p/'sensor.exit').read_text().strip()=='0'
 assert t[0]<=12 and len(src)-t[-1]<=15 and (np.diff(t)>0).all()
 assert (curr==src[t]).all() and (prev==src[t-1]).all()
 assert np.isfinite(co).all() and ((co>=0)&(co<=[384,448])).all()
 valid=np.ones(len(t),dtype=bool);delta=[]
 for a,b in ((0x1280e70,0x1280e7c),(0x1280e88,0x1280e94)):
  e=np.max(abs(.5*(pos(a)+pos(b))-co),axis=1)
  delta.append(float(e.max()));valid &= e<.02
 with gzip.open(p/'gray.raw.gz')as g:gray=np.frombuffer(g.read(),'u1').reshape(-1,112,96)
 assert len(gray)==len(t)
 rgb_t=[]
 for tick,bgr in color(p):
  rgb_t.append(tick);i=t.searchsorted(tick)
  sc=Image.fromarray(bgr[:,:,::-1]).crop((32,16,416,464)).convert('L').resize((96,112),Image.Resampling.BOX)
  assert np.array_equal(sc,gray[i]),'Native 640×480 BGR and policy camera failed byte identity'
 assert np.array_equal(np.array(rgb_t),t[f[:,3]==1])
 drift=support(t,src,co)
 # A hard rail may suppress a commanded movement; report per-axis all physically
 # measurable movement and recount mixed direction windows separately.
 j=np.flatnonzero((np.diff(t)==4)&(t[1:]+4<len(src))&valid[1:]&valid[:-1])+1
 bits=src[t[j]+1]
 direction=4+3*((bits&32>0).astype(int)-(bits&16>0).astype(int))+(bits&128>0).astype(int)-(bits&64>0).astype(int)
 bomb=(src&2>0);edge=np.r_[False,bomb[1:]&~bomb[:-1]]
 act=np.stack((direction,bits&4>0,bits&1>0,np.stack([edge[t[j]+k]for k in (1,2,3,4)]).any(0)),1).astype('u1')
 np.savez_compressed(p/'positions.npz',native_tick=t,xy_native=co,xy_camera=co/4,
    native_word=curr.astype('u2'),previous_native_word=prev.astype('u2'),full_color_ticks=np.array(rgb_t),
    causal_pair_rows=j,action_next_fourheads=act,stage_replay_words=src,xy_qualified=valid,stage=meta['stage'])
 return dict(**meta,original_replay_sha256=meta['source_sha256'],excluded_xy_ticks=t[~valid].tolist(),eligible_pairs_use_two_verified_poses=True,opening_xy=co[0].tolist(),closing_xy=co[-1].tolist(),
     rows=len(t),causal_four_tick_pairs=len(j),native_tick_range=[int(t[0]),int(t[-1])],native_source_words=len(src),
     observed_current_word_matches=int(sum(curr==src[t])),previous_word_matches=int(sum(prev==src[t-1])),
     full_color_witnesses=len(rgb_t),full_color_adjacent_pairs=int((np.diff(rgb_t)==4).sum()),
     full_color_to_policy_byte_identical=len(rgb_t),bank_selects_from_train_only=True,
     support_by_input_temporal_shift={str(k):v for k,v in drift.items()},body_box_midpoint_max_error_native_px=delta,
     xy_native_range=[co.min(0).tolist(),co.max(0).tolist()])
def templates(p):
 f,_,_,pos=xy(p);t=f[:,0];co=pos(0x1280e58)
 bank=[]
 for tk,bgr in color(p):
  if tk%240!=0 or tk>3600:continue
  x,y=np.rint(co[t.searchsorted(tk)]).astype(int)
  if not (12<=x<372 and 16<=y<430):continue
  roi=bgr[16+y-11:16+y+12,32+x-9:32+x+10,:].copy()
  if roi.shape!=(23,19,3):continue
  # Uniform templates from original TRAIN only. No test position is available to search.
  bank.append((tk,roi))
 assert len(bank)>=5
 return bank
def evaluate(p,bank):
 f,_,_,pos=xy(p);t=f[:,0];co=pos(0x1280e58);examples=[];pred=[];target=[];scores=[];ticks=[]
 thumbnails=[]
 for tk,bgr in color(p):
  if tk%240 or (p.name=='s1' and tk<=3600):continue
  field=np.ascontiguousarray(bgr[16:464,32:416,:])
  best=-2;pt=(0,0)
  for _,r in bank:
   _,v,_,loc=cv2.minMaxLoc(cv2.matchTemplate(field,r,cv2.TM_CCOEFF_NORMED))
   if v>best:best=v;pt=(loc[0]+9,loc[1]+11)
  c=co[t.searchsorted(tk)];pred.append(pt);target.append(c);ticks.append(tk);scores.append(best)
  if len(thumbnails)<10 and tk%960==0:
   im=Image.fromarray(field[:,:,::-1]);draw=ImageDraw.Draw(im)
   x,y=np.rint(c).astype(int)
   draw.rectangle((x-13,y-18,x+13,y+18),outline='lime',width=2)
   u,v=pt;draw.ellipse((u-5,v-5,u+5,v+5),outline='yellow',width=1)
   im.thumbnail((160,190));thumbnails.append((tk,im))
 acc=np.array(pred);a=np.array(target);err=np.linalg.norm(acc-a,axis=1);perm=np.linalg.norm(acc-np.roll(a,len(a)//2,axis=0),axis=1)
 sheet=Image.new('RGB',(len(thumbnails)//2*165+165,432),'white');d=ImageDraw.Draw(sheet)
 for ix,(tk,im) in enumerate(thumbnails):
  col=ix//2;row=ix%2;sheet.paste(im,(col*165,row*216+20));d.text((col*165,row*216+2),str(tk),fill='black')
 sheet.save(p/'camera-registration.jpg',quality=87)
 np.savez(p/'visual-check.npz',native_tick=ticks,matched_xy=acc,xy_native=a,ncc=scores)
 return dict(train_template_count=len(bank),train_template_ticks=[x for x,_ in bank],
     search='Frozen TRAIN 23x19 BGR templates; full-field normalized cross-correlation. The search receives no target coordinate.',
     frames=len(err),median_error_native_px=float(np.median(err)),p90_error_native_px=float(np.percentile(err,90)),
     within_12px=int((err<=12).sum()),within_24px=int((err<=24).sum()),shuffled_median_error_px=float(np.median(perm)),
     eval_source='Held unseen independent source' if p.name=='held' else 'same author TRAIN, different scene windows' if p.name in ('s1','s2') else 'TRAIN includes template source frames')

def do(p,meta,source):
 doc=verified(p,meta)
 bank=templates(source)
 doc['camera_registration']=evaluate(p,bank)
 doc['coverage_every_fourth_native_tick']=doc['rows']/(doc['native_tick_range'][1]/4)
 doc['artifacts_sha256']={f.name:sha(f) for f in sorted(p.iterdir()) if f.is_file() and f.suffix in ('.bin','.csv','.tsv','.npz','.gz','.png','.jpg','.sha256')}
 direction=doc['support_by_input_temporal_shift']['0']
 doc['movement_agreement_at_zero_shift']=direction
 doc['qualification']=bool(direction[0]>=100 and direction[1]>=.99 and doc['causal_four_tick_pairs']>100)
 save(p/'source-audit.json',doc)
 print(p,doc['rows'],doc['causal_four_tick_pairs'],doc['excluded_xy_ticks'],direction,doc['camera_registration']['median_error_native_px'],flush=True)
if __name__=='__main__':
 for arg in sys.argv[1:]:
  p=Path(arg).resolve()
  if p.name in ('s1','s2'):
   meta=json.loads((p/'capture.json').read_bytes());source=p.parent/'s1'
  else:
   meta=dict(stage=2,shot='reimua',rank='Normal',role=p.name,author='Boredperson' if p.name=='train' else 'zylo',source_sha256=SHA[p.name]);source=p.parent/'train'
  do(p,meta,source)
