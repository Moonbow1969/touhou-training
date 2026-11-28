"""Encode original captured RGB frames using their measured monotonic capture times."""
import csv
import json
from pathlib import Path
import subprocess
import sys

root=Path('/var/tmp/moonbow-th07-trial-261127')
mode=sys.argv[1]
if mode=='engineering':
    folder=root/'virtual'
    rows=list(csv.reader((folder/'neutral-tail.tsv').open(),delimiter='\t'))
    files=[folder/'neutral-tail'/f'{int(n):04d}.jpg' for n,_ in rows]
    times=[int(t)/1e9 for _,t in rows]
    name='neutral-tail-real-time.mp4'
else:
    folder=root/'zero'
    rows=list(csv.DictReader((folder/'controls.csv').open()))
    times=[float(r['t']) for r in rows]
    files=[folder/'movie'/f'{i:05d}.jpg' for i in range(len(rows))]
    times.extend([times[-1]+.12,times[-1]+1.20])
    from PIL import Image
    for name in ['settled-continue','settled-scorecard']:
        frame=folder/f'{name}.jpg'
        with Image.open(folder/f'{name}.png') as image:
            image.convert('RGB').crop((0,0,640,480)).save(frame,quality=94)
        files.append(frame)
    name='normal-zero-shot-original.mp4'
lines=['ffconcat version 1.0']
for i,f in enumerate(files):
    if not f.exists():raise FileNotFoundError(f)
    lines.extend([f"file '{f}'",f"duration {times[i+1]-times[i]:.6f}" if i+1<len(files) else 'duration 2.0'])
lines.append(f"file '{files[-1]}'")
manifest=folder/'video.ffconcat'
manifest.write_text('\n'.join(lines)+'\n')
subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','concat','-safe','0',
                '-i',str(manifest),'-vf','crop=640:480:0:0','-fps_mode','vfr','-c:v','libx264',
                '-threads','1','-preset','fast','-crf','24','-pix_fmt','yuv420p','-movflags','+faststart',
                str(folder/name)],check=True)
