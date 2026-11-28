"""Render the actual decision stream at measured wall time; no live-original recording overhead."""
from pathlib import Path
import argparse
import subprocess
import numpy as np
from PIL import Image,ImageDraw

def run(folder):
    with np.load(folder/"policy-trace.npz") as z:
        t=z["clock_s"];ticks=z["native_tick"];camera=z["frames"];actions=z["action"]
    order=np.asarray(t)-t[0]
    fps=15; count=int(np.ceil((order[-1]+.07)*fps))
    enc=subprocess.Popen(["ffmpeg","-hide_banner","-loglevel","error","-y","-f","rawvideo","-pix_fmt","rgb24","-s","640x480",
                          "-r",str(fps),"-i","pipe:0","-c:v","libx264","-threads","1","-crf","22","-pix_fmt","yuv420p",
                          "-movflags","+faststart",str(folder/"native-credit.mp4")],stdin=subprocess.PIPE)
    def emit(im,n=1):
        data=im.convert("RGB").tobytes()
        for _ in range(n):enc.stdin.write(data)
    emit(Image.open(folder/"full-stock-opening.png").crop((0,0,640,480)),fps)
    for i in np.searchsorted(order,np.arange(count)/fps,side="right")-1:
        im=Image.new("RGB",(640,480));im.paste(Image.fromarray(camera[i]).resize((384,448),Image.Resampling.NEAREST),(32,16))
        label=ImageDraw.Draw(im);label.text((430,82),f"TH07 0.11 Normal\nReimu A\n\nOriginal camera\n96 x 112 grayscale\n\nNative tick {ticks[i]}\n\nDirection {actions[i,0]}\nFocus {actions[i,1]}\nShot {actions[i,2]}\nBomb {actions[i,3]}\n\n1x wall playback",fill="white",spacing=5)
        emit(im)
    for file in ("settled-continue.png","settled-scorecard.png"):
        emit(Image.open(folder/file).crop((0,0,640,480)),2*fps)
    enc.stdin.close()
    if enc.wait():raise RuntimeError("ffmpeg failed")
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("folder",type=Path);run(parser.parse_args().folder)
