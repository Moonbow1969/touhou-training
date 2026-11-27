"""Serve 96 two-slot TH12 timing probes using one visual inference per four native ticks."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import time

import numpy as np
import torch

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--directory',type=Path,required=True);ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--sha',required=True);ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--mode',choices=('four-tick','two-subwindows'),required=True)
    ap.add_argument('--warmup-passes',type=int,default=0)
    a=ap.parse_args();assert hashlib.sha256(a.model.read_bytes()).hexdigest()==a.sha
    torch.set_num_threads(1);torch.manual_seed(a.seed)
    actor=torch.jit.load(str(a.model),map_location='cpu').eval()
    output=a.directory;output.mkdir(exist_ok=True);path=output/'actor.sock'
    if a.warmup_passes:
        if not 1 <= a.warmup_passes <= 16 or path.exists():
            raise ValueError('Warm-up requires a fresh private socket path')
        x=torch.zeros((1,2,112,96),dtype=torch.float32)
        rng=torch.get_rng_state()
        times=[]
        with torch.inference_mode():
            for _ in range(a.warmup_passes):
                start=time.perf_counter_ns()
                h=actor(x)
                assert [tuple(v.shape) for v in h]==[(1,2,9),(1,2,2),(1,2,2),(1,2,2)]
                assert all(v.isfinite().all() for v in h)
                times.append((time.perf_counter_ns()-start)/1e6)
        if not torch.equal(rng,torch.get_rng_state()):
            raise ValueError('Dry forward changed the behavior RNG')
        (output/'warmup.json').write_text(json.dumps(dict(
            policy_sha256=a.sha,forward_ms=times,shape=list(x.shape),input='float32 zeros',
            input_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(),
            rng_state_unchanged=True,socket_bound=False),indent=2)+'\n')
    frames=[];actions=[];logits=[];logp=[];prev=None
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server,torch.inference_mode():
        server.bind(str(path));server.listen(1)
        client,_=server.accept()
        with client:
            for _ in range(96):
                data=client.recv(112*96,socket.MSG_WAITALL)
                if len(data)!=112*96:raise ValueError('Incomplete original camera block')
                cam=np.frombuffer(data,np.uint8).reshape(112,96).copy()
                if prev is None:prev=cam
                inp=np.stack((cam,np.abs(cam.astype('i2')-prev).astype('u1')))
                heads=actor(torch.from_numpy(inp).unsqueeze(0).float()/255.)
                if [tuple(v.shape) for v in heads]!=[(1,2,9),(1,2,2),(1,2,2),(1,2,2)]:raise ValueError('Unknown model output')
                joint=[];lp=[]
                for slot in ([0] if a.mode=='four-tick' else [0,1]):
                    acts=[int(torch.multinomial(h[0,slot].softmax(-1),1)) for h in heads]
                    joint.append(acts);lp.append([float(h[0,slot].log_softmax(-1)[k]) for h,k in zip(heads,acts)])
                if a.mode=='four-tick':joint.append(joint[0].copy());lp.append(lp[0].copy())
                client.sendall(np.array(joint,dtype=np.uint8).tobytes())
                frames.append(cam);prev=cam;actions.append(joint);logp.append(lp);logits.append(torch.cat(heads,dim=-1)[0].numpy())
    np.savez_compressed(output/'model-wire.npz',frames=np.asarray(frames,dtype='u1'),action=np.asarray(actions,dtype='u1'),
                        logp=np.asarray(logp,dtype='f4'),logits=np.asarray(logits,dtype='f4'))
    path.unlink()
if __name__=='__main__':main()
