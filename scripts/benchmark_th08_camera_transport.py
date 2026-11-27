"""Measure local WSL camera transport on identical original observations."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import zstandard

from scripts.checkpoint_storage import digest
from scripts.run_th08_trial_camera import area_camera, linux, read_exact
from scripts.train_five_original_games import freeze


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('engineering','scratch','out'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True)
    pixel=[];crop=[];inputs=[]
    for seed in (1620801,1620802):
        movie=a.engineering/str(seed)/'original-observations.mp4'
        cap=cv2.VideoCapture(str(movie))
        indices=np.linspace(20,int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-2,12,dtype=int)
        inputs.append(dict(seed=seed,movie_sha256=digest(movie),decoded_frame_indices=indices.tolist()))
        for n in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(n));ok,frame=cap.read()
            if not ok:raise ValueError('Missing game movie fixture')
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            pixel.append(rgb.tobytes())
            gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
            crop.append(area_camera(rgb).tobytes()+rgb[22:226,430:636].tobytes()+gray[170:210,108:326].tobytes())
        cap.release()
    sets={'whole_rgb':pixel,'policy_camera_and_lossless_observers':crop}
    for name,rows in sets.items():
        (a.scratch/(name+'.bin')).write_bytes(b''.join(rows))
    trials=[]
    for rep in range(3):
        conditions=[(s,c) for s in sets for c in ('raw','zstd')]
        for i in np.random.default_rng(8191+rep).permutation(4):
            name,codec=conditions[i];rows=sets[name];size=len(rows[0]);fingerprints=[hashlib.sha256(x).digest() for x in rows]
            cmd=['wsl','-d','Ubuntu-26.04','--','docker','run','--rm','-i','--network','none','--cpus=.8','--memory=160m',
                 '--mount','type=bind,source='+linux(a.scratch)+',target=/fixtures,readonly','--entrypoint','/fixtures/bench-transport',
                 'toho-gamer-trial:v6','/fixtures/'+name+'.bin',str(size),codec]
            decoder=zstandard.ZstdDecompressor();samples=[]
            with subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE) as proc:
                for j in range(75):
                    ix=(j*7)%24;start=time.perf_counter_ns()
                    proc.stdin.write(bytes([ix]));proc.stdin.flush()
                    length,encode=map(int,proc.stdout.readline().split());wire=read_exact(proc.stdout,length)
                    unpack=time.perf_counter_ns()
                    array=decoder.decompress(wire,max_output_size=size) if codec=='zstd' else wire
                    finish=time.perf_counter_ns()
                    if hashlib.sha256(array).digest()!=fingerprints[ix]:raise ValueError('Non-identical decoded observation')
                    if j>=3:samples.append(dict(roundtrip_ms=(finish-start)/1e6,bytes=length,encoder_cpu_ms=encode/1e6,decoder_wall_ms=(finish-unpack)/1e6))
                proc.stdin.write(b'\xff');proc.stdin.flush();proc.stdin.close()
                if proc.wait(timeout=15):raise ValueError('Transport failure')
            results={k:float(np.median([s[k] for s in samples])) for k in samples[0]}
            results['p95_roundtrip_ms']=float(np.quantile([s['roundtrip_ms'] for s in samples],.95))
            results['mean_bytes']=float(np.mean([s['bytes'] for s in samples]))
            trials.append(dict(repeat=rep,representation=name,codec=codec,summary=results,samples=samples))
    freeze(a.out/'report.json',dict(hardware='Gamer Windows Python to its local Ubuntu 26.04 WSL2 Docker process, no network, 0.8 CPU cap on sender.',inputs=inputs,
           notes='The 24 fixture observations are decoded from the two complete original movies. Decoded bytes match between transport treatments. Benchmarked request/response, compression and decompression only. Native capture, actor inference, keyboard hold time, and remote LAN transport excluded. Compact mode retains the policy and observer crops and needs a worker-side movie encoder for full-color recordings.',trials=trials))


if __name__=='__main__':
    main()
