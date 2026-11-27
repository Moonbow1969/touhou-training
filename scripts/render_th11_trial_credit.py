"""Render an original TH11 camera credit on its logged wall clock and show its full-color endpoint."""
import argparse
import csv
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw
from scripts.render_th10_matched_credit import font,sha
from scripts.collect_th10_full_action import save

def run(folder):
    receipt=json.loads((folder/"receipt.json").read_bytes())
    assert sha(folder/"trajectory.npz")==receipt["trajectory_sha256"]
    assert sha(folder/"endpoint.png")==receipt["evidence_sha256"]["endpoint.png"]
    with np.load(folder/"trajectory.npz") as d:
        frames=d["frames"];t=(d["wall_ns"]-d["wall_ns"][0])/1e9;actions=d["action"]
    rows=list(csv.DictReader((folder/"behavior.csv").open()))
    post=np.array([int(x["after_ms"])-int(rows[0]["before_ms"]) for x in rows])/1000
    end=post[-1];rank=["Easy","Normal","Hard","Lunatic"][receipt["difficulty"]]
    final=Image.open(folder/"endpoint.png").convert("RGB").crop((3,22,643,502)).resize((320,240),Image.Resampling.LANCZOS)
    out=folder/"full-credit.mp4";fps=15;count=int(np.ceil((end+2)*fps))
    head=font(20,True);regular=font(16);small=font(13);direction=["up-left","up","up-right","left","neutral","right","down-left","down","down-right"]
    with imageio.get_writer(out,fps=fps,codec="libx264",quality=6,macro_block_size=2,output_params=["-threads","2","-movflags","+faststart"]) as video:
        for n in range(count):
            clock=n/fps;board=Image.new("RGB",(592,364),(14,23,32));draw=ImageDraw.Draw(board)
            draw.text((12,8),f"TH11   {rank}   Reimu A",font=head,fill="white")
            draw.text((12,38),f"Original 0.02a trial   |   seed {receipt['seed']}   |   1x playback",font=regular,fill="#c4d0d6")
            if clock<end:
                i=max(0,min(len(t)-1,int(np.searchsorted(t,clock,side="right")-1)))
                board.paste(Image.fromarray(frames[i]).resize((240,280),Image.Resampling.NEAREST),(12,70))
                a=actions[i];j=int(np.searchsorted(post,clock,side="right")-1)
                score,power,spares=(int(rows[j]["score"])*10,int(rows[j]["power"])/20,int(rows[j]["spares"])) if j>=0 else (0,0,2)
                entries=[f"Time     {clock:.1f} s",f"Score    {score:,}",f"Spares   {max(spares,0)}",f"Power    {power:.2f}",f"Move     {direction[a[0]]}",
                         "Keys      "+"  ".join(label for value,label in zip(a[1:],["focus","shoot","X"]) if value)]
                for k,label in enumerate(entries):draw.text((279,88+k*35),label,font=regular,fill="white")
                draw.text((279,321),"96 x 112 native camera",font=small,fill="#aabfc8")
            else:
                board.paste(final,(12,86))
                for k,label in enumerate(["Original Game Over",f"Score   {receipt['score']:,}","Spares  0",f"Power   {receipt['terminal_power_units']/20:.2f}","Continues  0"]):
                    draw.text((347,104+34*k),label,font=regular,fill="white")
            video.append_data(np.asarray(board))
    reader=imageio.get_reader(out)
    try:assert reader.count_frames()==count
    finally:reader.close()
    save(folder/"video.json",dict(fps=fps,frames=count,sha256=sha(out),source_trajectory_sha256=receipt["trajectory_sha256"],playback="Complete logged wall clock at 1x; two seconds of original Game Over screen after last native loss."))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("folder",type=Path);run(p.parse_args().folder)
