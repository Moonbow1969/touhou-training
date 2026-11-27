"""Synchronize two real original-client recordings to their first applied learned actions."""
from pathlib import Path
import argparse,hashlib,json,subprocess
import cv2
import imageio_ffmpeg
import numpy as np

def clip(folder):
    r=json.loads((folder/'receipt.json').read_bytes())
    meta=json.loads((folder/'original-client-recording.json').read_bytes())
    path=folder/meta['file']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('Modified original video')
    offset=(meta['actor_first_decision_ns']-meta['first_frame_ns'])/1e9
    stream=cv2.VideoCapture(str(path))
    return r,meta,offset,stream,cv2.imread(str(folder/'terminal.png'))
def render(a,b,out):
    clips=[clip(a),clip(b)]
    left,right=(c[0] for c in clips)
    if any(left[k]!=right[k] for k in ('seed','host','lane','difficulty')):raise ValueError('Different native pairing')
    fps=5;w,h=960,440
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}',
         '-r',str(fps),'-i','-','-an','-c:v','libx264','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(out)]
    writer=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    starts=[round(c[2]*fps) for c in clips]
    for c,start in zip(clips,starts):c[3].set(cv2.CAP_PROP_POS_FRAMES,start)
    count=int(np.ceil((max(c[0]['seconds'] for c in clips)+2)*fps))
    try:
        for j in range(count):
            t=j/fps;canvas=np.zeros((h,w,3),'u1');canvas[:]=(25,25,25)
            for side,(r,m,offset,capture,terminal) in enumerate(clips):
                x=side*480
                ended=t>=r['seconds']
                if ended:frame=terminal
                else:
                    ok,frame=capture.read()
                    if not ok:raise EOFError('Original client stream ended before credit')
                canvas[46:406,x:x+480]=cv2.resize(frame,(480,360),interpolation=cv2.INTER_AREA)
                label=('Four-game human SFT' if r['arm']=='four-sft' else 'Four-game SFT + PPO')
                cv2.putText(canvas,label,(x+9,32),cv2.FONT_HERSHEY_SIMPLEX,.7,(225,229,230),1,cv2.LINE_AA)
                status=f'Game Over  Stage {r["maximum_stage"]}  Last playable Bombs {r["last_live_bombs"]}' if ended else f'{t:.1f} s  {r["difficulty_name"]}  seed {r["seed"]}'
                cv2.putText(canvas,status,(x+9,430),cv2.FONT_HERSHEY_SIMPLEX,.48,(200,212,220),1,cv2.LINE_AA)
            writer.stdin.write(canvas.tobytes())
    finally:
        writer.stdin.close()
        for c in clips:c[3].release()
    if writer.wait():raise RuntimeError('Encoder failed')
    return dict(file=out.name,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),seed=left['seed'],host=left['host'],lane=left['lane'],
                fps=fps,frames=count,dimensions=[w,h],source_sha256=[c[1]['sha256'] for c in clips],
                timing='Aligned at first applied learned action. Each side holds its actual terminal screenshot after its own credit ends.')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['sft','ppo','video']:p.add_argument(k,type=Path)
    a=p.parse_args();result=render(a.sft,a.ppo,a.video)
    a.video.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
