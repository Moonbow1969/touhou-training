"""Audit per-pulse native latency and independently track the original rendered sprite."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def timing(values):
    if not values:return dict(n=0)
    return dict(n=len(values),min=float(np.min(values)),median=float(np.median(values)),
                p95=float(np.quantile(values,.95)),max=float(np.max(values)))


def visual_lag(roi, capture_ticks, native, crop):
    verified_tick=native[:,1]
    center_y=int(round(16+np.median(native[:,4])-crop[1]))
    red,green,blue=np.moveaxis(roi[:,center_y-10:center_y+11].astype(np.float32),-1,0)
    red_excess=np.maximum(0,red-np.maximum(green,blue)-18) * (red>=125)
    x_coordinates=np.arange(red.shape[2])
    target=np.interp(capture_ticks,verified_tick,native[:,3])+224-crop[0]
    weights=red_excess.sum(1)*(abs(target[:,None]-x_coordinates)<=27)
    live=weights.sum(1)>500
    observed=(weights@x_coordinates)/np.maximum(weights.sum(1),1)
    input_turn=abs(np.interp(capture_ticks+3,verified_tick,native[:,3])-np.interp(capture_ticks-3,verified_tick,native[:,3]))>=10
    live &= input_turn & (capture_ticks>=verified_tick[0]+9) & (capture_ticks<=verified_tick[-1]-9)
    if live.sum()<20:return dict(rows=int(live.sum()))
    shifts=[]
    for shift in range(-4,9):
        x=np.interp(capture_ticks[live]-shift,verified_tick,native[:,3])
        error=observed[live]-x
        deviation=np.abs(error-np.median(error))
        shifts.append(dict(lag_native_ticks=shift,mean_absolute_px=float(deviation.mean()),median_absolute_px=float(np.median(deviation))))
    best=min(shifts,key=lambda row:row['mean_absolute_px'])
    return dict(rows=int(live.sum()),best=best,by_lag=shifts)


def audit(source,crop):
    with np.load(source,allow_pickle=False) as raw:
        commands,native,frames,roi=(raw[k] for k in ('commands','native','frames','roi'))
    if not len(commands):return None
    pulses=[]
    for item in commands[commands[:,2]!=0]:
        pulse,scheduled,mask,sign,pre,post,issued,done,os_lr=map(int,item)
        release=commands[(commands[:,0]==pulse)&(commands[:,2]==0)]
        if len(release)!=1:continue
        before=native[native[:,1]<=pre]
        after=native[(native[:,1]>=pre)&(native[:,1]<=pre+9)]
        if not len(before) or not len(after):continue
        baseline=before[-1,3]
        key=after[(after[:,2].astype(int)&192)==mask]
        displaced=after[(after[:,3]-baseline)*sign>1]
        released=after[(after[:,1]>=release[0,4]) & ((after[:,2].astype(int)&192)==0)]
        if not len(key) or not len(displaced):continue
        consumed=int(key[0,1])
        move=int(displaced[0,1])
        pulses.append(dict(pulse=pulse,scheduled_tick=scheduled,requested=mask,issued_native_bracket=[pre,post],
            os_report_matches=(os_lr==1 if sign<0 else os_lr==2),native_consumed_tick=consumed,native_first_xy_tick=move,
            native_neutral_tick=int(released[0,1]) if len(released) else None,
            issue_to_key_ms=float((key[0,0]-issued)/1e6),send_call_ms=(done-issued)/1e6,
            issue_to_key_tick_interval=[consumed-post,consumed-pre],key_to_xy_ticks=move-consumed))
    return dict(source=source.parent.name,file_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              sent_pulses=int(np.count_nonzero(commands[:,2])),qualified_pulses=pulses,frames=len(frames),
              capture_span_ticks=(frames[:,3]-frames[:,1]).tolist(),
              capture_time_ms=((frames[:,2]-frames[:,0])/1e6).tolist(),
              native_tick_gaps=timing(np.diff(native[:,1]).tolist()),
              visual_track_by_capture_start=visual_lag(roi,frames[:,1],native,crop),
              visual_track_by_capture_end=visual_lag(roi,frames[:,3],native,crop))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('acquisition',type=Path)
    args=parser.parse_args()
    root=args.acquisition
    plan=json.loads((root/'protocol.json').read_bytes())
    rows=[x for file in sorted(root.glob('credit-*/trace.npz')) if (x:=audit(file,plan['crop_ltrb'])) is not None]
    pulse=[p for row in rows for p in row['qualified_pulses']]
    if not pulse:raise SystemExit('No eligible native transition pulse')
    result=dict(game=plan['game'],pulses=len(pulse),source_credits=len(rows),
                os_mask_matches=sum(p['os_report_matches'] for p in pulse),
                requested_to_consumed_lower_tick=timing([p['issue_to_key_tick_interval'][0] for p in pulse]),
                requested_to_consumed_upper_tick=timing([p['issue_to_key_tick_interval'][1] for p in pulse]),
                issue_to_first_observed_consumed_ms=timing([p['issue_to_key_ms'] for p in pulse]),
                native_key_to_xy_tick=timing([p['key_to_xy_ticks'] for p in pulse]),
                capture_duration_ms=timing([t for r in rows for t in r['capture_time_ms']]),
                captures_cross_native_tick=sum(sum(t!=0 for t in r['capture_span_ticks']) for r in rows),
                captures=sum(r['frames'] for r in rows),credits=rows,
                interpretation='Readback times are first observer samples. The rendered player estimate uses red sprite pattern and horizontal displacement during a fixed-Y early opening. This clock alignment does not time effects after capture or display photon latency.')
    (root/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='credits'},indent=2))
    for r in rows:
        print(r['source'],'visual lag by capture start/end:',
              r['visual_track_by_capture_start'].get('best'),r['visual_track_by_capture_end'].get('best'))
