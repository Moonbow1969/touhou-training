"""Render three complete, time-aligned TH10 policy-camera recordings and native Game Over captures."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size,bold=False):
    options=(('C:/Windows/Fonts/segoeuib.ttf','DejaVuSans-Bold.ttf') if bold else
             ('C:/Windows/Fonts/segoeui.ttf','DejaVuSans.ttf'))
    for file in options:
        try:return ImageFont.truetype(file,size)
        except OSError:pass
    return ImageFont.load_default()


def load(folder):
    r=json.loads((folder/'report.json').read_bytes())
    assert r['terminal'] and r['continues']==0 and r['final_power'] is not None
    assert sha(folder/'behavior.npz')==r['behavior_sha256']
    assert sha(folder/'endpoint.png')==r['terminal_screenshot_sha256']
    with np.load(folder/'original-pixels.npz') as z:
        frames=z['obs'];ts=(z['monotonic_ns']-z['monotonic_ns'][0])/1e9
    with np.load(folder/'behavior.npz') as z:start=z['action_utc_ms'][0]
    with gzip.open(folder/'native.csv.gz','rt') as f:rows=list(csv.DictReader(f))
    clocks=(np.array([int(row['utc_ms']) for row in rows])-start)/1000
    assert len(frames)==r['steps']+1 and all(np.diff(ts)>0)
    return dict(report=r,frames=frames,times=ts,states=rows,native_clocks=clocks,
                endpoint=Image.open(folder/'endpoint.png').convert('RGB').resize((288,216),Image.Resampling.LANCZOS),
                metadata=dict(policy_sha256=r['policy_sha256'],game_sha256=r['game_sha256'],build=r['source']['bundle'],
                    policy_camera_sha256=sha(folder/'original-pixels.npz'),behavior_sha256=r['behavior_sha256'],
                    endpoint_sha256=r['terminal_screenshot_sha256'],native_csv_gz_sha256=sha(folder/'native.csv.gz'),
                    saved_camera_frames=len(frames),original_shape=list(frames.shape[1:]),original_frame_delta_median_ms=float(np.median(np.diff(ts))*1000),
                    final_native_wall_s=float(clocks[-1]),endpoint=r['native_outcome'],seed=r['seed'],host=r['host']))


def render(args):
    data=[load(p) for p in args.arms]
    assert data[0]['report']['arm']=='sft' and data[1]['report']['arm'] in ('ppo','ppo1') and data[2]['report']['arm']=='ppo2'
    seed=data[0]['report']['seed']
    assert all(r['report']['seed']==seed and r['report']['host']==data[0]['report']['host'] for r in data)
    size=(960,508);fps=15
    count=int(np.ceil((max(r['native_clocks'][-1] for r in data)+2)*fps))
    labels=['Replay SFT','PPO update 1','PPO update 2']
    fonts=[font(21,True),font(15,True),font(14),font(12)]
    cache=[{} for _ in data];colors=['#8dc4e9','#ffbd88','#99d3ac']
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with imageio.get_writer(args.out,fps=fps,codec='libx264',quality=5,pixelformat='yuv420p',macro_block_size=2,
                            output_params=['-threads','2','-movflags','+faststart']) as video:
        for tick in range(count):
            t=tick/fps
            panel=Image.new('RGB',size,'#111922');draw=ImageDraw.Draw(panel)
            draw.text((18,8),f'TH10  |  Normal story  |  Reimu A  |  seed {seed}',font=fonts[0],fill='white')
            draw.text((18,39),'1x recorded wall time   /   original game under Wine   /   complete credits   /   no Continue',font=fonts[2],fill='#b9c8d6')
            for i,arm in enumerate(data):
                x=16+320*i;r=arm['report'];times=arm['times'];end=arm['native_clocks'][-1]
                draw.text((x,66),labels[i],font=fonts[1],fill=colors[i])
                draw.rectangle((x,91,x+287,426),fill='#080b0f')
                idx=max(0,int(np.searchsorted(times,t,side='right')-1))
                if t<end:
                    if idx not in cache[i]:
                        cache[i].clear()
                        cache[i][idx]=Image.fromarray(arm['frames'][idx]).convert('RGB').resize((288,336),Image.Resampling.NEAREST)
                    panel.paste(cache[i][idx],(x,91))
                    s=arm['states'][max(0,int(np.searchsorted(arm['native_clocks'],t,side='right')-1))]
                    left=f"Stage {s['stage']}    Spares {s['lives']}    Power {int(s['power_units'])/20:.2f}"
                    right=f'{t:5.1f} s'
                else:
                    panel.paste(arm['endpoint'],(x,143))
                    draw.text((x+8,381),'Original Game Over screen',font=fonts[2],fill=colors[i])
                    left=f"Game Over    Spares {r['final_lives']}    Power {r['final_power']:.2f}"
                    right=f'{end:5.1f} s'
                draw.text((x,436),left,font=fonts[2],fill='white')
                draw.text((x+239,436),right,font=fonts[2],fill=colors[i])
            draw.text((18,477),'Camera: captured 96 x 112 grayscale playfield, held between observations (~8 Hz). Game Over: original color screenshot.',
                      font=fonts[3],fill='#b9c8d6')
            video.append_data(np.asarray(panel))
    meta=dict(video=args.out.name,sha256=sha(args.out),rendered_frames=count,encoded_fps=fps,
              playback='1x original wall-time sample-and-hold; two extra seconds at the final Game Over screen. The 96x112 actor cameras were sampled around 8 Hz. Color terminal captures were taken after inventory settled.',
              recordings={name:a['metadata'] for name,a in zip(labels,data)})
    args.out.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(meta,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arms',nargs=3,type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);render(p.parse_args())
