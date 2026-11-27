"""Measure single-decision CPU forward latency on the original native camera stream."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch


def benchmark(camera,models):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    frames=np.fromfile(camera,np.uint8).reshape(-1,112,96)
    if len(frames)<129:raise ValueError('At least 129 native observations are required')
    indices=np.linspace(1,len(frames)-1,128,dtype=int)
    gray=torch.from_numpy(frames[indices]).float()/255
    previous=torch.from_numpy(frames[indices-1]).float()/255
    inputs=torch.stack((gray,(gray-previous).abs()),1)
    actors=[torch.jit.load(str(p),map_location='cpu').eval() for p in models]
    latency=[[],[]]
    with torch.inference_mode():
        for actor in actors:
            for frame in inputs[:18]:actor(frame[None])
        for repeat in range(3):
            for j,frame in enumerate(inputs):
                for index in ((0,1) if (j+repeat)%2 else (1,0)):
                    start=time.perf_counter_ns();actors[index](frame[None])
                    latency[index].append((time.perf_counter_ns()-start)/1e6)
    return dict(threads=1,batch_size=1,torch_version=torch.__version__,native_frames=len(frames),
        source_sha256=hashlib.sha256(camera.read_bytes()).hexdigest(),
        arms=[dict(policy_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),n=len(values),
                   p50_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),mean_ms=float(np.mean(values)))
              for path,values in zip(models,latency)])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--camera',type=Path,required=True)
    p.add_argument('--policies',type=Path,nargs=2,required=True)
    args=p.parse_args();print(json.dumps(benchmark(args.camera,args.policies),indent=2))
