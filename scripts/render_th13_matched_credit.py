"""Render complete original TH13 camera credits with native stocks and terminal ranking photos."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from scripts.collect_th10_full_action import save
from scripts.render_th10_matched_credit import font, sha


NAMES = {
    "base-cnn":"14-player CNN", "base-attention":"14-player attention",
    "new-cnn":"38-player CNN", "new-attention":"38-player attention",
}
DIRECTIONS = ("up-left","up","up-right","left","neutral","right","down-left","down","down-right")


def render(native, seed, output):
    clips=[]
    for arm, label in NAMES.items():
        folder=native/f"held-{arm}-{seed}"
        r=json.loads((folder/"receipt.json").read_bytes())
        assert r["seed"]==seed and r["arm"]==arm and r["host"]=="accelerator" and r["terminal"] and not r["continues"]
        states=np.fromfile(folder/"steps.bin","<f4").reshape(-1,143)
        frames=np.fromfile(folder/"observations.u8",np.uint8).reshape(-1,112,96)
        assert len(states)==len(frames)==r["steps"] and states[-1,120] and not states[:-1,120].any()
        assert sha(folder/"rollout.npz")==r["sha256"] and np.all(np.diff(states[:,121])>0)
        clips.append(dict(arm=arm,label=label,r=r,states=states,frames=frames,
            start=np.r_[0.,states[:-1,121]],end=float(states[-1,121]),
            endpoint=Image.open(folder/"ranking.png").convert("RGB").crop((0,0,640,480)).resize((216,162),Image.Resampling.LANCZOS),
            sources={name:sha(folder/name) for name in ("steps.bin","observations.u8","ranking.png","receipt.json","rollout.npz")}))
    output.parent.mkdir(parents=True,exist_ok=True)
    fps=15;count=int(np.ceil((max(c["end"] for c in clips)+2)*fps))
    head=font(22,True);title=font(16,True);text=font(14);small=font(12)
    with imageio.get_writer(output,fps=fps,codec="libx264",quality=5,pixelformat="yuv420p",macro_block_size=2,
                            output_params=["-threads","2","-movflags","+faststart"]) as video:
        for n in range(count):
            clock=n/fps
            panel=Image.new("RGB",(1008,424),"#111a22");draw=ImageDraw.Draw(panel)
            draw.text((15,8),f"TH13   ·   Normal story   ·   controller seed {seed}",font=head,fill="white")
            draw.text((15,37),"Original trial 0.01a on Accelerator   ·   full starting stock   ·   no Continue   ·   1× game recording",font=text,fill="#c1ced6")
            for i,c in enumerate(clips):
                x=15+250*i
                draw.text((x,68),c["label"],font=title,fill=("#9eccec","#86c7b6","#f4b47a","#cfb2ed")[i])
                draw.rectangle((x,96,x+215,347),fill="#080d11")
                if clock<c["end"]:
                    j=max(0,int(np.searchsorted(c["start"],clock,side="right")-1))
                    frame=Image.fromarray(c["frames"][j]).resize((216,252),Image.Resampling.NEAREST)
                    panel.paste(frame,(x,96));s=c["states"][j]
                    move,focus,shoot,bomb=map(int,s[110:114])
                    a=f"Stage {int(s[125])}    Spares {int(s[127])}    Bombs {int(s[129])}"
                    buttons=[DIRECTIONS[move]]+[label for on,label in ((focus,"focus"),(shoot,"shoot"),(bomb,"Bomb")) if on]
                    b="  ".join(buttons)
                else:
                    panel.paste(c["endpoint"],(x,134))
                    outcome=c['r']['outcome'].replace('_',' ').title()
                    a=f"{outcome}    Bombs {c['r']['remaining_bombs']}"
                    b=f"Original ranking at {c['end']:.1f} s"
                draw.text((x,355),a,font=text,fill="white");draw.text((x,378),b,font=small,fill="#b0bdc7")
            draw.text((15,402),"Each panel shows the original 96 × 112 agent camera and sampled controls. Final color frames are the original game's score rankings.",
                      font=small,fill="#c1ced6")
            video.append_data(np.asarray(panel))
    save(output.with_suffix(".json"),dict(seed=seed,host="accelerator",fps=fps,frames=count,sha256=sha(output),
         playback="Full credits, original native wall-clock intervals at 1×, 2 seconds of final rankings. Camera frames precede each sampled action; start times use the end time of the previous logged action interval.",
         credits={c["arm"]:dict(steps=c["r"]["steps"],duration_s=c["end"],policy_sha256=c["r"]["policy_sha256"],
              endpoint=c["r"]["outcome"],game_sha256=c["r"]["game_sha256"],source_sha256=c["sources"]) for c in clips}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("native13",type=Path);p.add_argument("--seed",type=int,required=True)
    p.add_argument("--out",type=Path,required=True);args=p.parse_args();render(args.native13,args.seed,args.out)
