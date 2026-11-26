"""Render three original playfield observation streams beside their logged controls."""
import argparse
from pathlib import Path
import subprocess

import imageio_ffmpeg
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode",action="append",type=Path,required=True)
    parser.add_argument("--label",action="append",required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if len(args.episode)!=3 or len(args.label)!=3:
        parser.error("Supply three episode paths and three labels in the same order.")
    streams=[]
    for p in args.episode:
        with np.load(p,allow_pickle=False) as z:
            ns=z["monotonic_ns"].astype(np.int64)
            streams.append((z["obs"],z["actions"],(ns-ns[0])*1e-9))
    width,height,fps=886,418,20
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),"-nostdin","-v","error","-n","-f","rawvideo","-pix_fmt","rgb24",
         "-s",f"{width}x{height}","-r",str(fps),"-i","pipe:0","-an","-c:v","libx264","-crf","20",
         "-pix_fmt","yuv420p","-movflags","+faststart",str(args.out)]
    header=ImageFont.load_default(size=19)
    detail=ImageFont.load_default(size=15)
    end=min(times[-1] for _,_,times in streams)
    with subprocess.Popen(cmd,stdin=subprocess.PIPE) as encoder:
        for tick in range(int(end*fps)+1):
            t=tick/fps
            im=Image.new("RGB",(width,height),"#f4f3ee")
            draw=ImageDraw.Draw(im)
            for col,((obs,acts,times),label) in enumerate(zip(streams,args.label)):
                x=5+col*294
                i=np.searchsorted(times,t,side="right")-1
                draw.text((x,5),label,font=header,fill="#19373a")
                frame=Image.fromarray(obs[i]).resize((288,336),Image.Resampling.NEAREST)
                im.paste(frame,(x,30))
                if i<len(acts):
                    dx,dy,shoot,focus=map(int,acts[i])
                    keys=[k for k,on in (("left",dx<0),("right",dx>0),("up",dy<0),("down",dy>0),
                                         ("shoot",shoot),("focus",focus)) if on]
                    draw.text((x,370)," + ".join(keys) or "release",font=detail,fill="#19373a")
                draw.text((x,392),f"Observation {i:02d}   |   elapsed {times[i]:.2f} s",font=detail,fill="#445859")
            encoder.stdin.write(im.tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("Encoding recorded observations failed.")


if __name__=="__main__":
    main()
