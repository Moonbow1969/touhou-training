"""Drive a frozen four-head policy on its isolated retail Game Start slot."""
import csv
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import time
import numpy as np
import torch
from PIL import Image

root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tools'))
from menu import cmd,key,snap
seed=int(sys.argv[1])
if len(sys.argv)==3:
    benchmark=json.loads((root/'record/held-protocol.json').read_bytes())
    arm=sys.argv[2]
    assert seed in benchmark['primary_seeds']+benchmark['extension_seeds']+benchmark['technical_reserves']
    plan=dict(seed=seed,arm=arm,sha256=benchmark['models'][arm],role='held')
else:
    plan=next(x for x in json.loads((root/'record/protocol.json').read_bytes())['lanes'] if x['seed']==seed)
out=root/'record'/f"{seed}-{plan['arm']}"
name=(out/'actual-container').read_text().strip() if (out/'actual-container').exists() else f'moonbow-th07-retail-record-{seed}'
policy=Path('/var/tmp/moonbow-th07-trial-261127/policy/policy.ts') if plan['arm']=='parent' else root/'retail-sft/fit/71281/policy.ts'
assert hashlib.sha256(policy.read_bytes()).hexdigest()==plan['sha256']
assert json.loads((out/'ready.json').read_bytes())==plan
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
model=torch.jit.load(str(policy),map_location='cpu').eval()
generator=torch.Generator().manual_seed(seed)
with torch.inference_mode():
    for _ in range(4):assert [a.numel() for a in model(torch.zeros(1,2,112,96))[:4]]==[9,2,2,2]
proc=subprocess.Popen(['docker','exec','-i',name,'/tool/record_actor'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
def exact(n):
    b=proc.stdout.read(n)
    if len(b)!=n:raise RuntimeError(f'Native actor short read {len(b)}/{n}, return {proc.poll()}')
    return b
frames=[];actions=[];logs=[];times=[];ticks=[];values=[];prev=None
start=time.monotonic()
# Enter ordinary play from the preverified shot selector. The private replay directory is empty.
cmd(name,'test -z "$(ls -A /tmp/th07/replay)" && xdotool keydown z; sleep .075; xdotool keyup z')
try:
    with (out/'controls.csv').open('x',newline='',buffering=1) as f, torch.inference_mode():
        wr=csv.writer(f);wr.writerow(('seq','tick','requested','spares_hud','bombs_hud','camera_ns','wall_s'))
        while True:
            kind,i,tick,word,spares,bombs,ns=struct.unpack('<6IQ',exact(32))
            assert i==len(actions)
            if kind in (2,3):break
            assert kind==1 and tick%4==0
            image=np.frombuffer(exact(112*96),'u1').copy().reshape(112,96)
            x=torch.from_numpy(image).float()/255
            obs=torch.stack((x,torch.zeros_like(x) if prev is None else (x-prev).abs()))[None]
            heads=model(obs)
            action=[int(torch.multinomial(h.softmax(-1),1,generator=generator)) for h in heads[:4]]
            move,focus,shoot,bomb=action
            mask=(16,0,32)[move//3]|(64,0,128)[move%3]|(4*focus)|shoot|(2*bomb)
            proc.stdin.write(struct.pack('<I',mask));proc.stdin.flush()
            latency=time.monotonic()-start
            wr.writerow((i,tick,mask,spares,bombs,ns,latency))
            frames.append(image);actions.append(action);ticks.append(tick);times.append(ns)
            logs.append([float(h.log_softmax(-1)[0,v])for h,v in zip(heads[:4],action)])
            values.append(float(heads[4].item())if len(heads)>4 else 0.)
            prev=x
            if i==0:subprocess.Popen(['docker','exec',name,'scrot','-o','/capture/full-opening.png'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    proc.stdin.close()
    if proc.wait(timeout=5)!=0:raise RuntimeError('Native worker failed to complete')
finally:
    cmd(name,'xdotool keyup Left Right Up Down Shift_L z x')
    np.savez_compressed(out/'policy-trace.npz', frames=np.asarray(frames), action=np.asarray(actions),
                        logp=np.asarray(logs),value=np.asarray(values),native_tick=np.asarray(ticks),camera_ns=np.asarray(times))
time.sleep(1.3);snap(name,'native-ending' if kind==3 else 'native-continue')
(out/'finished-game.json').write_text(json.dumps({**plan,'actions':len(actions),'last_tick':ticks[-1],
                                                 'terminal_observed_tick':tick,'outcome':'ending' if kind==3 else 'game-over',
                                                 'wall_seconds':time.monotonic()-start})+'\n')
print('NATIVE TERMINAL',seed,len(actions),tick,flush=True)
