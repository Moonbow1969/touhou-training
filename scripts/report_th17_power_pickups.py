"""Describe attainable native power changes and compare them to one original HUD witness."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools/rollout-worker-probe/th17-full"))
from timeline import initialized_stock_rows, progressed_ticks

def inspect(path,expected_split='train'):
    r=json.loads((path/'receipt.json').read_bytes())
    if r['split']!=expected_split or hashlib.sha256((path/'native.tsv').read_bytes()).hexdigest()!=r['native_trace_sha256']:raise ValueError(path)
    z=np.genfromtxt(path/'native.tsv',names=True,delimiter='\t',dtype=None,encoding='utf8')
    z,_=initialized_stock_rows(z[z['stage']>=1],r.get('observer_version',1))
    elapsed=progressed_ticks(z['stage'],z['tick'])
    last_loss=None;gain=0;near=0;transitions=0;near_transitions=0;downward_units=0;post_loss_decreases=0
    for i in range(1,len(z)):
        current,prior=z[i],z[i-1]
        delta=int(current['power'])-int(prior['power'])
        if current['spares']<prior['spares']:
            last_loss=elapsed[i]
        elif current['spares']==prior['spares'] and current['spares']>=0 and current['stage']==prior['stage'] and prior['power']>=100 and delta>0:
            gain+=delta;transitions+=1
            if last_loss is not None and elapsed[i]-last_loss<=240:
                near+=delta;near_transitions+=1
        if delta<0:
            downward_units-=delta
            if last_loss is not None and elapsed[i]-last_loss<=15:post_loss_decreases-=delta
    initial=100;peak=int(z['power'][z['spares']>=0].max())
    return dict(seed=r['seed'],rank=r['difficulty'],policy=r['policy_sha256'],native_sha256=r['native_trace_sha256'],positive_same_life_power_transitions=transitions,
                peak_displayed_power=peak/100,unique_record_units=peak-initial,nonrecord_units=gain-(peak-initial),native_ticks=int(elapsed[-1]),
                positive_power_units=gain,units_in_first_240_native_ticks_after_loss=near,transitions_in_post_loss_window=near_transitions,
                downward_power_units=downward_units,downward_units_in_first_15_ticks_after_loss=post_loss_decreases,
                max_life_pieces=int(z['life_pieces'].max()),max_bomb_pieces=int(z['bomb_pieces'].max()))

def native_hud(path,out):
    import cv2
    video=json.loads((path/'original-client-recording.json').read_bytes())
    if hashlib.sha256((path/video['file']).read_bytes()).hexdigest()!=video['sha256']:raise ValueError('Original recording changed')
    trace=np.genfromtxt(path/'decisions.tsv',names=True,delimiter='\t',dtype=None,encoding='utf8')
    native=np.genfromtxt(path/'native.tsv',names=True,delimiter='\t',dtype=None,encoding='utf8');native=native[native['stage']==1]
    capture=cv2.VideoCapture(str(path/video['file']));sheet=np.full((248,740,3),25,'u1');witness=[]
    for j,tick in enumerate((1000,1100)):
        i=int(np.argmin(np.abs(trace['tick']-tick)))
        native_tick=int(trace['tick'][i]);native_power=int(native['power'][np.searchsorted(native['tick'],native_tick,side='right')-1])
        frame=int(round((int(trace['onset_ns'][i])-video['first_frame_ns'])*video['frames_per_second']/1.e9))
        capture.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,img=capture.read()
        if not ok:raise EOFError('Missing original source frame')
        patch=img[165:235,430:625]
        x=10+j*370
        sheet[60:184,x:x+350]=cv2.resize(patch,(350,124),interpolation=cv2.INTER_NEAREST)
        cv2.putText(sheet,f'Original Stage 1, tick {native_tick}',(x,37),cv2.FONT_HERSHEY_SIMPLEX,.62,(235,235,235),1,cv2.LINE_AA)
        cv2.putText(sheet,f'Native power {native_power/100:.2f}, frame {frame}',(x,219),cv2.FONT_HERSHEY_SIMPLEX,.58,(235,235,235),1,cv2.LINE_AA)
        witness.append(dict(frame=frame,native_stage_tick=native_tick,native_power=native_power))
    capture.release();cv2.imwrite(str(out),sheet)
    return dict(seed=int(path.name.rsplit('-',1)[1]),original_recording_sha256=video['sha256'],independent_original_hud_file=out.name,
                evidence_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),frames=witness)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('native',type=Path);p.add_argument('output',type=Path);p.add_argument('--hud-witness',type=Path);a=p.parse_args()
    protocol=json.loads((a.native.parent/'camera-protocol.json').read_bytes())
    rows=[inspect(f.parent) for f in sorted(a.native.glob('train-*/receipt.json'))]
    if len(rows)!=len(protocol['train_seeds']['17']) or {r['seed'] for r in rows}!=set(protocol['train_seeds']['17']):raise ValueError('Incomplete training bank')
    counts=('positive_power_units','positive_same_life_power_transitions','units_in_first_240_native_ticks_after_loss','downward_power_units','downward_units_in_first_15_ticks_after_loss')
    by_rank={}
    for rank in (2,3):
        rr=[r for r in rows if r['rank']==rank]
        by_rank[rank]=dict(credits=len(rr),credits_with_same_life_gain=sum(r['positive_power_units']>0 for r in rr),
                           credits_with_nonzero_life_fragments=sum(r['max_life_pieces']>0 for r in rr),credits_with_nonzero_bomb_fragments=sum(r['max_bomb_pieces']>0 for r in rr),
                           **{key:sum(r[key] for r in rr) for key in counts})
    out=dict(by_rank=by_rank,credits=rows,scope='Training-only, native power in hundredths of one displayed Power. A positive observation may combine multiple pickups. The post-loss time window is descriptive and does not identify whether a particular pickup was a death drop. No rewards were modified.')
    if a.hud_witness:
        out['hud_witness']=native_hud(a.hud_witness,a.output.with_name('power-hud-original.png'))
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(by_rank,indent=2))

if __name__=='__main__':main()
