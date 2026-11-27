"""Private per-checkpoint TorchScript inference for TH10 container-local Unix sockets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socketserver
import struct
import time
import torch


def heads(model,frame,previous):
    difference=torch.zeros_like(frame) if previous is None else (frame-previous).abs()
    logits=model(torch.stack((frame,difference)).unsqueeze(0))
    if not isinstance(logits,(tuple,list)) or len(logits) not in (4,5) or [tuple(v.shape) for v in logits[:4]]!=[(1,9),(1,2),(1,2),(1,2)]:
        raise ValueError("Expected four categorical logits with widths 9/2/2/2.")
    value=logits[4].reshape(1,1) if len(logits)==5 else frame.new_zeros(1,1)
    result=torch.cat([*logits[:4],value],1).flatten()
    if not torch.isfinite(result).all():raise ValueError("Nonfinite camera policy.")
    return result.tolist()


class Camera(socketserver.StreamRequestHandler):
    def handle(self):
        self.request.settimeout(3600)
        self.request.sendall(self.server.sha.encode('ascii'))
        previous=None;elapsed=[];count=0
        with torch.inference_mode():
            while True:
                raw=self.rfile.read(112*96)
                if not raw:break
                if len(raw)!=112*96:raise EOFError("Incomplete camera frame.")
                started=time.perf_counter()
                frame=torch.frombuffer(bytearray(raw),dtype=torch.uint8).reshape(112,96).float()/255
                logits=heads(self.server.model,frame,previous);previous=frame
                self.request.sendall(struct.pack('<16f',*logits));count+=1
                elapsed.append((time.perf_counter()-started)*1000)
        if count:print(json.dumps(dict(decisions=count,mean_service_ms=sum(elapsed)/count,
                                      maximum_service_ms=max(elapsed))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--sha',required=True)
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--socket-name',default='actor.sock')
    a=p.parse_args()
    if Path(a.socket_name).name!=a.socket_name:raise ValueError("Specify a socket basename.")
    if hashlib.sha256(a.model.read_bytes()).hexdigest()!=a.sha:raise ValueError("Camera checkpoint changed")
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    model=torch.jit.load(str(a.model),map_location='cpu').eval()
    with torch.inference_mode():heads(model,torch.zeros(112,96),None)
    os.umask(0o077);os.chdir(a.directory)
    class Server(socketserver.ThreadingUnixStreamServer):
        daemon_threads=True
    with Server(a.socket_name,Camera) as server:
        server.sha=a.sha;server.model=model
        print(json.dumps(dict(ready=True,sha=a.sha,torch=torch.__version__,backend='CPU')),flush=True)
        server.serve_forever()
