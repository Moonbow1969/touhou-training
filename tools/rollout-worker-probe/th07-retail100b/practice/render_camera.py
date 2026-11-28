"""Render every captured policy image with original stock at the same native tick."""
import argparse,csv,hashlib,json
from pathlib import Path
import cv2,numpy as np

def run(source,output):
    end=json.loads((source/'finished-game.json').read_bytes())
    q=json.loads((source/'qualification.json').read_bytes())
    with np.load(source/'policy-trace.npz',allow_pickle=False) as z:
        frames,actions,ticks=[z[k] for k in ('frames','action','native_tick')]
    native={}
    with (source/'native-input.csv').open() as handle:
        for row in csv.DictReader(handle):native[int(row['tick'])]=row
    assert len(ticks)>1 and np.all(np.diff(ticks)==4)
    directions=['UP LEFT','UP','UP RIGHT','LEFT','STILL','RIGHT','DOWN LEFT','DOWN','DOWN RIGHT']
    writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),15,(690,384))
    if not writer.isOpened():raise RuntimeError('Video writer unavailable')
    for image,a,t in zip(frames,actions,ticks):
        view=np.full((384,690,3),(29,24,21),dtype=np.uint8)
        view[24:360,12:300]=cv2.cvtColor(cv2.resize(image,(288,336),interpolation=cv2.INTER_NEAREST),cv2.COLOR_GRAY2BGR)
        row=native[int(t)]
        lines=[('TH07 / PERFECT CHERRY BLOSSOM',.64,(240,225,196)),
               ('Normal Reimu A / Letty nonspell 1',.54,(224,221,220)),
               ('From official practice menu',.52,(185,189,186)),
               (f'Native tick   {t:5d} / {end["terminal_observed_tick"]}',.57,(203,212,206)),
               (f'Spare lives   {row["lives"]}     Bombs   {row["bombs"]}',.59,(140,220,223)),
               (f'Move   {directions[int(a[0])]}',.59,(235,224,197)),
               (f'Focus {a[1]}      Shot {a[2]}      Bomb {a[3]}',.56,(215,197,231)),
               ('96 x 112 original policy camera',.52,(197,197,197)),
               ('Sampled once per 4 native ticks',.52,(197,197,197)),
               ('TRAIN / '+str(end['seed']),.52,(174,177,183))]
        for (line,size,color),y in zip(lines,(39,72,98,155,189,239,269,313,337,369)):
            cv2.putText(view,line,(317,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)
        writer.write(view)
    final=cv2.imread(str(source/'native-continue.png'))
    if final is not None:
        panel=np.zeros((384,690,3),dtype=np.uint8)
        panel[3:363,10:490]=cv2.resize(final[:480,:640],(480,360))
        for y,line in [(112,'GAME OVER'),(157,'0 spare lives'),(184,'0 Bombs'),(216,'No Continue'),(355,'Original final screen')]:
            cv2.putText(panel,line,(500,y),cv2.FONT_HERSHEY_SIMPLEX,.46,(235,235,235),1,cv2.LINE_AA)
        for _ in range(45):writer.write(panel)
    writer.release()
    cap=cv2.VideoCapture(str(output))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==len(ticks)+45
    cap.release()
    print(output,output.stat().st_size,hashlib.sha256(output.read_bytes()).hexdigest())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
 a=p.parse_args();run(a.source,a.output)
