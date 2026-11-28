"""Measure the same frozen policy's post-capture CPU processing cost on observed original frames."""
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch

root=Path(__file__).resolve().parents[1]/'outputs/touhou-vision-261126/th10-retail-normal'
clip=cv2.VideoCapture(str(root/'six-ppo-101001-3/whole-credit.mp4'))
frames=[]
try:
    for i in range(300):
        ok,frame=clip.read()
        if not ok:break
        frames.append(frame[16:464,32:416].copy())
finally:
    clip.release()
if len(frames)!=300:raise ValueError('Expected 300 complete real gameplay frames')
torch.set_num_threads(1)
model=torch.jit.load(str(root/'normal-sft/policy.ts'),map_location='cpu').eval()
with torch.inference_mode():
    for i in range(30):
        model(torch.zeros(1,2,112,96))
    times=[]
    old=None
    for bgr in frames:
        start=time.perf_counter_ns()
        rgb=np.ascontiguousarray(bgr[:,:,::-1])
        gray=np.asarray(Image.fromarray(rgb).convert('L').resize((96,112),Image.Resampling.BOX))
        diff=np.abs(gray.astype('i2')-old.astype('i2')).astype('u1') if old is not None else np.zeros_like(gray)
        image=torch.from_numpy(np.stack((gray,diff)))[None].float().div_(255)
        heads=model(image)[:4]
        _=[head[0].softmax(0).numpy() for head in heads]
        end=time.perf_counter_ns()
        times.append((end-start)/1e6)
        old=gray
result=dict(frames=len(times),threads=1,unit='ms',median=float(np.median(times)),p95=float(np.quantile(times,.95)),
            max=float(max(times)),scope='Offline measured CPU time from a predecoded original playfield BGR image through color conversion, fourfold resize, pixel difference, model inference and probabilities. Includes no game capture, I/O or display delay.')
(root/'normal-sensor-diagnostic/post-capture-cpu-cost.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result))
