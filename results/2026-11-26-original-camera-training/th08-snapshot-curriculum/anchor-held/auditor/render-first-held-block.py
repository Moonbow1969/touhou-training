"""Render the first registered held block from its exact policy-camera shards."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
from matplotlib.font_manager import findfont


def render(directory,out):
    arms=[]
    for s in (80965501,80965502):
        row=json.loads((directory/f"attempts/{s}.json").read_bytes())
        assert row["time_block"]==1 and row["physical_endpoint_verified"]
        file=directory/row["file"]
        assert hashlib.sha256(file.read_bytes()).hexdigest()==row["sha256"]
        with np.load(file,allow_pickle=False)as archive:
            t={k:archive[k] for k in archive.files}
        assert np.all(t["native_ticks_under_command"]==4) and row["exact_windows"]==len(t["action"])
        arms.append((row,t))
    arms.sort(key=lambda pair:pair[0]["arm"]!="lambda1-parent")
    fonts=[ImageFont.truetype(findfont("DejaVu Sans"),size) for size in (17,14,12)]
    n=max(len(a["action"]) for _,a in arms)
    command=[imageio_ffmpeg.get_ffmpeg_exe(),"-v","error","-y","-f","rawvideo","-pix_fmt","rgb24",
             "-s","688x480","-r","15","-i","pipe:0","-c:v","libx264","-preset","fast","-crf","23",
             "-pix_fmt","yuv420p","-movflags","+faststart",str(out)]
    encoder=subprocess.Popen(command,stdin=subprocess.PIPE)
    direction=("up-left","up","up-right","left","neutral","right","down-left","down","down-right")
    colors=("#9cdee3","#f2bd8d")
    for i in range(n+16):
        image=Image.new("RGB",(688,480),"#171d22")
        draw=ImageDraw.Draw(image)
        draw.text((17,8),"TH08 Normal Ghost  |  First held block",font=fonts[0],fill="white")
        draw.text((17,33),"Actual 96 x 112 grayscale policy cameras at one decision per four native ticks.",font=fonts[2],fill="#d0d8df")
        for j,(row,trace) in enumerate(arms):
            x=17+j*339
            count=len(trace["action"])
            playing=i<count
            frame=trace["frames"][i] if playing else trace["boundary_frame"]
            nearest=Image.fromarray(frame).resize((288,336),Image.Resampling.NEAREST).convert("RGB")
            image.paste(nearest,(x,81))
            name="Frozen parent" if j==0 else "Curriculum PPO"
            draw.text((x,57),f"{name}  |  {row['seed']}",font=fonts[1],fill=colors[j])
            if playing:
                move,focus,shot,bomb=map(int,trace["action"][i])
                commands="  ".join(v for v,k in (("focus",focus),("Z",shot),("X",bomb)) if k)
                caption=f"{direction[move]}   {commands}"
                tick=int(trace["native_tick"][i]);stock=[int(trace["spares"][i]),int(trace["bombs"][i])]
            else:
                tick=int(trace["native_next_tick"][-1]);stock=row["end_stock"]
                caption="First spare loss - final observed camera held"
            draw.text((x,425),f"Stage 1 tick {tick}    Spares {stock[0]}    Bombs {stock[1]}",font=fonts[1],fill="white")
            draw.text((x,450),caption,font=fonts[2],fill=colors[j])
        encoder.stdin.write(np.asarray(image,dtype=np.uint8).tobytes())
    encoder.stdin.close()
    if encoder.wait()!=0:
        raise RuntimeError("Failed to encode the held policy-camera movie")
    metadata=dict(seeds=[r["seed"] for r,_ in arms],arms=[r["arm"] for r,_ in arms],
                  source_sha256=[r["sha256"]for r,_ in arms],source="Original 96x112 grayscale action cameras",
                  native_ticks_per_decision=4,playback_fps=15,hold_last_endscreen=True,
                  total_video_frames=n+16,bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    out.with_suffix(".json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("study",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    render(args.study,args.output)
