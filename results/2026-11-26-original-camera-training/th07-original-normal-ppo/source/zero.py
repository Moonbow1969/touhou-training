"""One isolated original TH07 Normal credit, shared four-head CPU TorchScript camera."""
from pathlib import Path
from hashlib import sha256
import json, subprocess, time, csv, os
import numpy as np
from PIL import Image
import torch

base=Path('/var/tmp/moonbow-th07-trial-261127')
out=Path(os.environ.get('TH07_OUT',base/'zero'))
model=Path(os.environ.get('TH07_MODEL',base/'policy/policy.ts'))
expected=os.environ.get('TH07_SHA','a6e1dec1625af50e214f141a4e405795fbdae905d78c4b887e423de0bca6a77e')
assert sha256(model.read_bytes()).hexdigest()==expected
seed=int(os.environ.get('TH07_SEED','2070701'))
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
actor=torch.jit.load(str(model),map_location='cpu').eval()
generator=torch.Generator().manual_seed(seed)
container=os.environ.get('TH07_CONTAINER','moonbow-th07-011-zero-2070701')
rpc=subprocess.Popen(['docker','exec','-i',container,'sh'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
def command(cmd):
    rpc.stdin.write(cmd+" && echo COMPLETE\n");rpc.stdin.flush()
    if rpc.stdout.readline().strip()!='COMPLETE':raise RuntimeError(cmd)
def read_image(path):
    with Image.open(path) as img:return img.convert('RGB').crop((0,0,640,480))
def snap(name):
    command('scrot -o /capture/'+name+'.bmp')
    return read_image(out/(name+'.bmp'))
def press_z():
    command('xdotool keydown z; sleep .09; xdotool keyup z; sleep .65')
def textmask(image):
    arr=np.asarray(image.crop((114,175,306,203)).convert('L'))
    return arr>218
rep=textmask(read_image(base/'virtual/continue-default-no.png'))
def similarity(arr):
    return (arr&rep).sum()/max(1,(arr|rep).sum())
def template_similarity(img,path,box):
    a=np.asarray(img.crop(box).convert('L').resize((90,35)),dtype=float).flatten()
    b=np.asarray(read_image(path).crop(box).convert('L').resize((90,35)),dtype=float).flatten()
    a-=a.mean();b-=b.mean()
    return a.dot(b)/max(1,np.linalg.norm(a)*np.linalg.norm(b))

for attempt in range(120):
    if (out/'window').is_file():break
    time.sleep(.5)
else:raise RuntimeError('Original window was not available after 60 seconds')
for attempt in range(40):
    img=snap('probe-menu')
    if template_similarity(img,base/'virtual/identity.png',(25,45,410,172))>.85:break
    if attempt in (8,22):command('xdotool keydown Escape; sleep .08; xdotool keyup Escape; sleep .6')
    time.sleep(.25)
else:raise RuntimeError('Original title mismatch')
img.save(out/'title.png')
for n,box in [(1,(214,190,438,262)),(2,(36,394,248,449)),(3,(361,190,589,261))]:
    press_z()
    img=snap('probe-menu')
    val=template_similarity(img,base/f'virtual/menu{n}.png',box)
    img.save(out/f'menu-{n}.png')
    if val<.66:raise RuntimeError(f'Menu scene {n} mismatch: {val}')
with torch.inference_mode():
    initial=actor(torch.zeros((1,2,112,96)))
    assert len(initial) in (4,5) and [x.numel()for x in initial[:4]]==[9,2,2,2]
command('xdotool keydown z; sleep .09; xdotool keyup z')
def stock(img):
    a=np.asarray(img,dtype=np.int16)
    counts=[]
    for y,life in [(103,True),(120,False)]:
        count=0
        for x in range(504,632,16):
            p=a[y-6:y+7,x-6:x+7];r,g,b=p.transpose(2,0,1)
            mask=((r>180)&(r-b>30)) if life else ((b>130)&(b-r>35))
            if mask.sum()<24:break
            count+=1
        counts.append(count)
    return counts
for _ in range(45):
    img=snap('live')
    if stock(img)==[2,3]:break
    time.sleep(.045)
else:raise RuntimeError('No native 2-spare 3-Bomb opening')
img.save(out/'full-stock-opening.png')
frames=[];actions=[];logs=[];values=[];times=[]
movie=out/'movie';movie.mkdir(exist_ok=True)
keys=set();prev=None;t0=time.monotonic()
allkeys=['Left','Right','Up','Down','Shift_L','z','x']
stop=False
try:
    with (out/'controls.csv').open('w',newline='') as file, torch.inference_mode():
        writer=csv.writer(file);writer.writerow(['t','move','focus','shot','bomb','spares','bombs','continue_template_iou'])
        for i in range(5000):
            t=time.monotonic()
            if i:img=snap('live')
            sim=similarity(textmask(img))
            if sim>.72:
                img.save(out/'native-continue.png');stop=True;break
            gray=np.asarray(img.crop((32,16,416,464)).convert('L').resize((96,112),Image.Resampling.BOX)).copy()
            x=torch.from_numpy(gray).float()/255
            obs=torch.stack([x,torch.zeros_like(x) if prev is None else (x-prev).abs()])[None]
            heads=actor(obs)
            a=[int(torch.multinomial(z.softmax(-1),1,generator=generator).item())for z in heads[:4]]
            lp=[float(z.log_softmax(-1)[0,v])for z,v in zip(heads[:4],a)]
            move,focus,shoot,bomb=a
            # Order is a 3x3 keypad, top row up-left/up/up-right, neutral=4.
            requested=set()
            if move%3!=1:requested.add('Left' if move%3==0 else 'Right')
            if move//3!=1:requested.add('Up' if move//3==0 else 'Down')
            requested.update(k for k,v in [('Shift_L',focus),('z',shoot),('x',bomb)] if v)
            cmds=[f'xdotool keyup {k}'for k in sorted(keys-requested)]+[f'xdotool keydown {k}'for k in sorted(requested-keys)]
            if cmds:command(' && '.join(cmds))
            keys=requested;prev=x
            frames.append(gray);actions.append(a);logs.append(lp);values.append(float(heads[4].item())if len(heads)==5 else 0)
            times.append(t-t0);writer.writerow([t-t0,*a,*stock(img),round(sim,4)]);file.flush()
            img.save(movie/f'{i:05d}.jpg',quality=80)
            time.sleep(max(0,.12-(time.monotonic()-t)))
finally:
    command('xdotool keyup '+' '.join(allkeys))
    np.savez_compressed(out/'policy-trace.npz',frames=np.asarray(frames),action=np.asarray(actions),logp=np.asarray(logs),value=np.asarray(values),clock_s=np.asarray(times))
if not stop:raise RuntimeError('Credit did not reach qualified original Continue scene')
time.sleep(.4)
img=snap('live')
if similarity(textmask(img))<.6:raise RuntimeError('Original Continue did not persist after key release')
img.save(out/'settled-continue.png')
# The original Continue prompt selects No by default, validated in the engineering credit.
if template_similarity(img,base/'virtual/continue-default-no.png',(183,270,269,311))<.75:
    raise RuntimeError('Original No selection not positively matched')
command('xdotool keydown z; sleep .1; xdotool keyup z; sleep 2')
for i in range(3):
    img=snap('live')
    if template_similarity(img,base/'virtual/settled-scorecard.png',(53,125,192,299))>.78:break
    img.save(out/f'official-ranking-{i}.png')
    command('xdotool keydown z; sleep .10; xdotool keyup z; sleep 1.3')
else:raise RuntimeError('Original terminal statistics not observed')
img.save(out/'settled-scorecard.png')
for i in range(6):
    if (out/'official-policy.rpy').exists():break
    command('xdotool keydown z; sleep .10; xdotool keyup z; sleep .80')
    copy=subprocess.run(['docker','cp',container+':/tmp/th07/replay/th7_01.rpy',str(out/'official-policy.rpy')],capture_output=True)
    if copy.returncode==0:break
else:raise RuntimeError('Original replay was not saved')
arr=np.asarray(actions)
(out/'receipt.pending-visual.json').write_text(json.dumps(dict(seed=seed,policy_sha256=expected,actions=len(actions),duration_s=times[-1],
    directions=np.bincount(arr[:,0],minlength=9).tolist(),button_active=arr[:,1:].sum(0).tolist(),
    median_step_ms=1000*float(np.median(np.diff(times))),first_stage='Stage 1',screen_only=True)))
rpc.stdin.close()
