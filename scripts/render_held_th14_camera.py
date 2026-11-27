"""Render one complete matched TH14 credit per checkpoint at a shared native-clock speed."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from scripts.checkpoint_storage import digest
from scripts.render_th10_matched_credit import font


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    runs=[]
    for arm in ('sft','ppo1','ppo2'):
        dir=a.root/f'v26th14-{arm}-{a.seed}';r=json.loads((dir/'report.json').read_bytes())
        if r['continued'] or r['seed']!=a.seed or digest(dir/'native.npz')!=r['trace_sha256']:raise ValueError('Unverified credit')
        with np.load(dir/'native.npz',allow_pickle=False) as z:
            images=z['frames'];states=z['native'];actions=z['action']
        clocks=states[:,0]-states[0,0]
        end=dir/'settled.png' if (dir/'settled.png').is_file() else dir/'terminal.png'
        with Image.open(end) as im:photo=im.crop((0,0,650,514)).convert('RGB').resize((288,228),Image.Resampling.LANCZOS)
        runs.append((r,images,states,clocks,actions,photo))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    bold=font(19,True);small=font(14);fps=15
    dirs=['up-left','up','up-right','left','neutral','right','down-left','down','down-right']
    with imageio.get_writer(a.out,fps=fps,codec='libx264',quality=6,macro_block_size=2,
                            output_params=['-threads','2','-movflags','+faststart']) as video:
        for tick in range(0,int(max(v[0]['elapsed_native_ticks'] for v in runs))+120,4):
            panel=Image.new('RGB',(936,510),'#101b25');draw=ImageDraw.Draw(panel)
            draw.text((12,8),f'Unseen TH14  |  Normal Stage 1  |  controller seed {a.seed}',font=bold,fill='white')
            draw.text((12,38),f'Original trial cameras, 60 native ticks/s playback  |  tick {tick:,}',font=small,fill='#bbd4da')
            for n,(r,images,states,clocks,actions,photo) in enumerate(runs):
                x=12+310*n;draw.text((x,65),['Human replay SFT','CNN PPO update 1','CNN PPO update 2'][n],font=bold,fill='#a2d4bc')
                if tick<r['elapsed_native_ticks']:
                    i=max(0,int(np.searchsorted(clocks,tick,side='right')-1))
                    frame=Image.fromarray(images[i]).resize((288,336),Image.Resampling.NEAREST).convert('RGB')
                    panel.paste(frame,(x,94));s=states[i];y=actions[i]
                    draw.text((x,440),f'Stage {int(s[2])} tick {int(s[1]):,}   |   {int(s[3])} spares   {int(s[4])} Bombs',font=small,fill='white')
                    buttons=' '.join(label for label,val in zip(('focus','shoot','Bomb'),y[1:]) if val)
                    draw.text((x,469),f'{dirs[y[0]]}   {buttons}',font=small,fill='#b4c3cb')
                else:
                    panel.paste(photo,(x,146))
                    draw.text((x,440),f'Game Over   |   0 spares   {r["terminal_bombs"]} Bombs',font=small,fill='white')
            video.append_data(np.asarray(panel))
    reader=imageio.get_reader(a.out);first=reader.get_data(0);n=reader.count_frames();reader.close()
    if n<30 or first.shape!=(510,936,3):raise ValueError('Encoded playback verification failed')
    a.out.with_suffix('.json').write_text(json.dumps(dict(seed=a.seed,frames=n,fps=fps,sha256=digest(a.out),
        source_credits=[dict(arm=v[0]['arm'],policy=v[0]['policy'],trace_sha256=v[0]['trace_sha256'],
                             native_ticks=v[0]['elapsed_native_ticks']) for v in runs]),indent=2)+'\n')


if __name__=='__main__':main()
