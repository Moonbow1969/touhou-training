"""Render the prospectively selected original-camera excerpts from the held TH08 AWR comparison."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SOURCES = [('awr',80966001,'Advantage weighted'),('uniform',80966003,'Uniform weight'),('lambda1',80966005,'Prior PPO')]
FPS = 15


def load(path):
    return json.loads(path.read_bytes())


def checksum(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def font(size):
    for face in ('DejaVuSans.ttf','C:/Windows/Fonts/segoeui.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        try:return ImageFont.truetype(face,size)
        except OSError:continue
    return ImageFont.load_default()


def source(study,arm,seed):
    root=study/arm;raw=root/'raw'/str(seed);summary=load(root/'summaries'/f'{seed}.json');receipt=root/'sealed'/f'{seed}.json';r=load(receipt)
    if checksum(receipt)!=summary['receipt_sha256'] or checksum(root/'sealed'/r['file'])!=r['sha256']:raise ValueError('Sealed example changed')
    q=load(root/'resource'/f'{seed}.json')
    if q['packed_sha256']!=r['sha256'] or summary['lane']!='A':raise ValueError('The preregistered comparison requires source lane A')
    with np.load(root/'sealed'/r['file'],allow_pickle=False) as z:
        imgs=z['frames'];acts=z['action'];ticks=z['native_tick'];life=z['spares'];bomb=z['bombs']
    rows=[json.loads(row) for row in (raw/'timing.jsonl').read_bytes().splitlines()]
    times=np.asarray([r['monotonic_ns'] for r in rows if r['phase']=='policy'][:len(imgs)],dtype=np.int64)
    if len(times)!=len(imgs) or not np.all(np.diff(times)>0):raise ValueError('Camera source registration failed')
    times=(times-times[0])/1e9;first=q['events'][0]['decision'] if q['events'] else len(imgs)-1
    starts=[0.,max(0.,times[first]-2.5),max(0.,times[-1]-4.)]
    return dict(frames=imgs,actions=acts,ticks=ticks,life=life,bomb=bomb,duration=times[-1],time=times,starts=starts,
                end=Image.open(raw/'terminal.png').convert('RGB'),source=summary['receipt_sha256'],summary=summary)


def main(study,out):
    frames=[source(study,k,s) for k,s,_ in SOURCES]
    width,height=936,570;colors=['#9ae6be','#b4d2fc','#eec78b'];large=font(25);medium=font(17);small=font(14)
    seq=[('Opening',0,5.),('First loss',1,4.),('Final controlled seconds',2,4.),('Original Game Over',3,2.)]
    sink=imageio_ffmpeg.write_frames(str(out),(width,height),fps=FPS,codec='libx264',quality=8,macro_block_size=2,ffmpeg_log_level='error');sink.send(None)
    dirs=['UL','UP','UR','LEFT','STILL','RIGHT','DL','DOWN','DR']
    try:
        for segment,part,duration in seq:
            for f in range(round(duration*FPS)):
                canvas=Image.new('RGB',(width,height),'#101820');draw=ImageDraw.Draw(canvas)
                draw.text((15,10),'TH08 Normal  /  Ghost Team',font=large,fill='white');draw.text((15,48),segment,font=medium,fill='#cbd9df')
                for i,(arm,seed,title) in enumerate(SOURCES):
                    data=frames[i];left=16+i*306;draw.text((left,84),title,font=medium,fill=colors[i]);draw.text((left,111),f'Controller seed {seed}',font=small,fill='#b5c0ca')
                    if part==3:
                        img=data['end'].resize((288,216),Image.Resampling.LANCZOS);canvas.paste(img,(left,190));stage=data['summary']['actual_endpoint']['stage'];draw.text((left,450),f'Final stage {stage or "unknown"}    Remaining stock',font=small,fill='white');stock=data['summary']['terminal_stock'];draw.text((left,476),f'{stock[0]} spare lives     {stock[1]} Bomb',font=medium,fill=colors[i]);continue
                    t=min(data['duration'],data['starts'][part]+f/FPS);idx=max(0,int(np.searchsorted(data['time'],t,side='right')-1))
                    img=Image.fromarray(data['frames'][idx]).resize((288,336),Image.Resampling.NEAREST).convert('RGB');canvas.paste(img,(left,138))
                    m,focus,fire,bomb=data['actions'][idx];keys=[dirs[m]]+(['SHIFT'] if focus else [])+(['Z'] if fire else [])+(['X'] if bomb else [])
                    draw.text((left,480),f"t={data['time'][idx]:.2f}s    tick {data['ticks'][idx]}    {' '.join(keys)}",font=small,fill='white');draw.text((left,505),f"{data['life'][idx]} spare lives   {data['bomb'][idx]} Bomb",font=small,fill=colors[i]);
                draw.text((15,542),'96 x 112 actor cameras sampled every four native ticks. Replay rendered at 15 fps using original timestamps.',font=small,fill='#a6b9c2');sink.send(np.asarray(canvas).tobytes())
    finally:sink.close()
    return {k:frames[i]['source'] for i,(k,_,_) in enumerate(SOURCES)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('study',type=Path);p.add_argument('out',type=Path);args=p.parse_args();print(main(args.study,args.out))
