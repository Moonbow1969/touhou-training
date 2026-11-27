"""Render every policy observation and the verified outcome in one original retail credit."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from scripts.checkpoint_storage import digest
from scripts.render_th10_matched_credit import font


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--credit',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    root=a.credit.parent
    r=json.loads(a.credit.read_bytes())
    if any(digest(root/f)!=r[k] for f,k in [('trajectory.npz','trace_sha256'),('opening.png','initial_image_sha256'),('endpoint.png','end_image_sha256')]):
        raise ValueError('Camera trace or original screenshot differs')
    with np.load(root/'trajectory.npz',allow_pickle=False) as z:
        frames,actions,native=z['frames'],z['actions'],z['native']
    n=len(frames)
    if not n==len(actions)==len(native)==r['decisions'] or frames.shape[1:]!=(112,96):raise ValueError('Original camera trace incomplete')
    a.out.parent.mkdir(exist_ok=True,parents=True)
    text,heading=font(16),font(20,True)
    movement=('up left','up','up right','left','still','right','down left','down','down right')
    with imageio.get_writer(a.out,fps=15,codec='libx264',quality=7,macro_block_size=1) as encoder:
        for i in range(n+25):
            panel=Image.new('RGB',(580,500),'#17202a')
            draw=ImageDraw.Draw(panel)
            draw.text((12,8),'TH14 retail 1.00b · Hard · Reimu A',font=heading,fill='white')
            if i<n:
                tick,stage,lives,bombs=native[i,1:5]
                image=Image.fromarray(frames[i]).resize((384,448),Image.Resampling.NEAREST)
                panel.paste(image,(10,39))
                labels=(f'Stage {stage}',f'Script tick {tick}',f'{max(0,lives)} spares',f'{bombs} Bombs',
                        movement[actions[i,0]],' '.join(k for k,on in zip(('Focus','Shot','Bomb'),actions[i,1:]) if on),
                        f'{i+1:,}/{n:,}', 'policy decisions')
                for j,s in enumerate(labels):draw.text((405,55+j*33),s,font=text,fill='white')
            else:
                with Image.open(root/'endpoint.png') as im:
                    end=im.convert('RGB')
                    end.thumbnail((520,400))
                    panel.paste(end,(14,40))
                draw.text((14,450),f"{r['terminal'].replace('_',' ')}  ·  {r['terminal_spares']} spares  ·  {r['terminal_bombs']} Bombs",font=heading,fill='white')
            encoder.append_data(np.asarray(panel))
    with imageio.get_reader(a.out) as reader:
        if reader.count_frames()!=n+25 or reader.get_data(0).shape!=(500,580,3):raise ValueError('Video verification failed')
    a.out.with_suffix('.json').write_text(json.dumps(dict(sha256=digest(a.out),credit_sha256=digest(a.credit),fps=15,
            frames=n+25,policy_decisions=n,alignment='All original decision cameras in order, one frame per decision, followed by the original terminal screenshot.'),indent=2)+'\n')


if __name__=='__main__':main()
