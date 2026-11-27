"""Reconstruct the complete camera and actual model controls of one stock TH17 credit."""
from pathlib import Path
import argparse,csv,json
import cv2
import numpy as np

def render(episode,target):
    receipt=json.loads((episode/'receipt.json').read_bytes())
    trace=np.genfromtxt(episode/'decisions.tsv',delimiter='\t',names=True)
    with np.load(episode/'behavior.npz') as z:camera=z['frames'];actions=z['action']
    time=(trace['onset_ns']-trace['onset_ns'][0])/1e9
    duration=(trace['observed_ns'][-1]-trace['onset_ns'][0])/1e9
    fps=8;size=(640,480);video=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),fps,size)
    if not video.isOpened():raise RuntimeError('MP4 writer unavailable')
    dirs=('up-left','up','up-right','left','neutral','right','down-left','down','down-right')
    for tick in range(int(np.ceil(duration*fps))):
        idx=max(0,min(len(actions)-1,int(np.searchsorted(time,tick/fps,side='right')-1)))
        frame=np.full((480,640,3),18,'u1');view=cv2.resize(camera[idx],(384,448),interpolation=cv2.INTER_NEAREST)
        frame[16:464,8:392]=cv2.cvtColor(view,cv2.COLOR_GRAY2BGR)
        lines=[('TH17  '+receipt['difficulty_name'],1),('Native 0.01a',.55),(receipt['arm'],.65),('Seed '+str(receipt['seed']),.54),(f'{tick/fps:.1f} s',.6),
               (f'Stage {int(trace["stage"][idx])}   Tick {int(trace["tick"][idx])}',.55),(f'Spares {int(trace["spares_before"][idx])}   Bombs {int(trace["bombs_before"][idx])}',.59),
               (dirs[actions[idx,0]],.72),('   '.join(name for name,v in zip(['focus','shoot','bomb'],actions[idx,1:]) if v),.56)]
        for j,(text,sz) in enumerate(lines):
            cv2.putText(frame,text,(402,44+j*43),cv2.FONT_HERSHEY_SIMPLEX,sz,(210,220,240),1,cv2.LINE_AA)
        video.write(frame)
    terminal=cv2.imread(str(episode/'terminal.png'))
    for _ in range(3*fps):video.write(terminal)
    video.release()
    probe=cv2.VideoCapture(str(target));probe.set(cv2.CAP_PROP_POS_FRAMES,int(probe.get(cv2.CAP_PROP_FRAME_COUNT))-1)
    assert probe.read()[0];probe.release()
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('episode',type=Path);p.add_argument('output',type=Path);a=p.parse_args();render(a.episode,a.output)
