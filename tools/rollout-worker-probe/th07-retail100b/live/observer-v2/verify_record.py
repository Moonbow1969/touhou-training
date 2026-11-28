"""Validate a complete retail original credit against independently saved and observed input."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from record_proof import verify_archive,official_score

root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tools'))
from seal import source_info

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def verify(folder,visual):
    plan=json.loads((folder/'finished-game.json').read_bytes())
    native=np.genfromtxt(folder/'native-input.csv',delimiter=',',names=True,dtype=None)
    ctl=np.genfromtxt(folder/'controls.csv',delimiter=',',names=True,dtype=None)
    ap=np.genfromtxt(folder/'applied.csv',delimiter=',',names=True,dtype=None)
    assert np.all(native['rank']==1) and np.all(native['shot']==0) and np.all(native['continues']==0)
    assert (native[0]['lives'],native[0]['bombs'])==(2,3) and native[-1]['retry']>0
    assert native[-1]['lives']==0 and native[-1]['misses']>=3
    assert np.all(native['event_index']==native['tick']) and np.max(np.abs(native['stageclock']-native['tick']))<=1
    stage_ids=sorted(set(native['stage'].tolist()))
    words={}
    for s in stage_ids:
        words[s],snapshot,score=source_info(folder/'official-policy.rpy',s)
        assert s>0
        archive=(folder/f'record-stage{s}.bin').read_bytes()
        memory=np.frombuffer(archive[44:],'<u2').reshape(-1,2)[:,0]
        length=min(len(memory),len(words[s]))
        assert length>0 and abs(len(memory)-len(words[s]))<=4
        assert np.array_equal(memory[:length],words[s][:length])
        assert np.array_equal(np.frombuffer(archive[:44],'u1')[32:40],snapshot[32:40])
    retained={}
    live_sets={}
    for entry in native:
        s,t=int(entry['stage']),int(entry['tick'])
        if (entry['flags']&4) and entry['event_index']==t and entry['word']==entry['record_slot']:
            live_sets.setdefault((s,t),set()).add(int(entry['word'])&247)
        if not entry['retry'] and (entry['flags']&4):
            retained[s,t]=(int(entry['word'])&247,int(entry['previous_word'])&247)
    offsets={}
    for off in range(-2,3):
        checks=[(word==(int(words[s][t+off])&247),prev==(int(words[s][t+off-1])&247))
                for (s,t),(word,prev) in retained.items() if 1<=t+off<len(words[s])]
        offsets[off]=np.mean(checks,axis=0).tolist()
    phase=max(offsets,key=lambda off:sum(offsets[off]))
    unique_phase=sum(offsets[phase])>max(sum(x) for off,x in offsets.items() if off!=phase)
    assert len(ctl)==len(ap)==plan['actions']
    with np.load(folder/'policy-trace.npz')as trace:
        a=trace['action'];ticks=trace['native_tick'];assert np.array_equal(ticks,ctl['tick'])
        assert np.array_equal(ap['camera_tick'],ticks) and np.array_equal(ctl['requested'],ap['requested_word'])
        masks=np.array([16,0,32])[a[:,0]//3]|np.array([64,0,128])[a[:,0]%3]|(4*a[:,1])|a[:,2]|(2*a[:,3])
        assert np.array_equal(masks,ctl['requested'])
        assert len(trace['frames'])==len(trace['logp'])==len(trace['value'])==len(a)
    # Assign actions using native real stage at camera timestamp, including stage-clock resets.
    ix=np.searchsorted(native['ns'],ctl['camera_ns'],side='right')-1
    assert np.all(ix>=0)
    action_stages=native['stage'][ix]
    stale=[];partial=[];bad=[];missing=0;total=0;exact=0
    for i,(nt,stage,mask) in enumerate(zip(ticks,action_stages,masks)):
        s,t,w=map(int,(stage,nt,mask))
        end=int(np.max(native['tick'][native['stage']==s]))
        ts=np.arange(t+1,min(t+4,end)+1)
        if len(ts)<4:
            assert i==len(ticks)-1 or action_stages[i+1]!=s
            partial.append([i,len(ts)])
        if not len(ts):continue
        saved=words[s][ts+phase]&247
        direct=any(w in live_sets.get((s,nt),set()) for nt in ts)
        if not (np.any(saved==w) and direct):stale.append(i)
        if len(ts)<4:continue
        gap=int((ticks[i+1]-t)//4)-1 if i+1<len(ticks) and action_stages[i+1]==s else 0
        assert gap>=0
        missing+=gap
        total+=1
        if np.all(saved==w) and gap==0:exact+=1
        else:bad.append([i,t,w,saved.tolist(),gap])
    ratio=exact/(total+missing)
    elapsed=(int(native[-1]['ns'])-int(native[0]['ns']))/1e9
    deaths=[]
    for i in np.flatnonzero(np.diff(native['misses'])>0)+1:
        deaths.append(dict(stage=int(native[i]['stage']),tick=int(native[i]['tick']),
                           prior_lives=int(native[i-1]['lives']),prior_bombs=int(native[i-1]['bombs'])))
    assert len(deaths)==native[-1]['misses']
    read_max=float((folder/'observer-timing.txt').read_text().strip().split('=')[1])/1e6
    freeze=(ap['resume_ns']-ap['freeze_start_ns'])/1e6
    freeze_to_x=(ctl['camera_ns']-ap['freeze_start_ns'])/1e6
    assert np.all(freeze_to_x>=0) and np.all(freeze>=freeze_to_x)
    assert np.all(ap['camera_tick']==ticks)
    archive=verify_archive(folder)
    try:
        score_proof=official_score(folder,score,int(native[-1]['score'])*10)
    except (ValueError,OSError) as exc:
        score_proof=dict(score_confirmed=False,error=str(exc),saved_rpy=int(score),last_live=int(native[-1]['score'])*10)
    import csv
    with (folder/'observer-distribution.csv').open() as f: timing=next(csv.DictReader(f))
    assert read_max==int(timing['max_ns'])/1e6
    p999=int(timing['p999_upper_ns'])/1e6
    passed=unique_phase and min(offsets[phase])>=.995 and ratio>=.995 and not stale and read_max<100 and np.all(ticks==ap['applied_tick']) and visual and p999<30 and all(n>0 for i,n in partial)
    result=dict(**plan,passed=bool(passed),archive=archive,score_proof=score_proof,observer_p999_upper_ms=p999,read_distribution=timing,recorded_stage_keys='official-policy.rpy',record_offset=phase,
                native_current_previous_match_by_offset=offsets,stage1_native_ticks=int(native['tick'][native['stage']==1].max()),
                stage2=2 in stage_ids,stage2_resources=None,stage1_deaths=sum(d['stage']==1 for d in deaths),
                deaths=deaths,actual_bombs=int(native[-1]['bombs_used']),terminal_stock=[int(native[-1]['lives']),int(native[-1]['bombs'])],
                original_score_ten_times_native=int(native[-1]['score'])*10,original_saved_score=score,
                terminal_continues=int(native[-1]['continues']),terminal_visually_confirmed=visual,
                native_observed_rows=len(native),complete_windows=total,strict_exact=exact,skipped_decisions=missing,
                nonexact=bad,stale=stale,partial=partial,exact_fraction=ratio,max_synchronous_observer_ms=read_max,
                max_freeze_ms=float(freeze.max()),p99_freeze_ms=float(np.quantile(freeze,.99)),native_wall_seconds=elapsed,
                max_freeze_to_x_ms=float(freeze_to_x.max()),p99_freeze_to_x_ms=float(np.quantile(freeze_to_x,.99)),
                timing_criterion="Registered criterion uses native acquisition calls. X display wall latency and total paused host time are reported separately.",
                camera_start_tick=int(ticks[0]),stage1_native_ticks_per_wall_second=int(native['tick'][native['stage']==1].max())/elapsed,
                artifacts_sha256={k:sha(folder/k) for k in ('full-opening.png','native-continue.png','settled-scorecard.png','native-input.csv','controls.csv','applied.csv','policy-trace.npz','official-policy.rpy','observer-timing.txt','observer-distribution.csv','events.bin','archive.sealed','settled-scorecard-repeat.png')})
    if result['stage2']:
        row=native[native['stage']==2][0]
        result['stage2_resources']=[int(row['lives']),int(row['bombs'])]
    (folder/'verified.json').write_text(json.dumps(result,indent=2)+'\n')
    print(plan['seed'],plan['arm'],'passed',passed,'phase',phase,offsets[phase],'exact',ratio,'stale',stale[:5],
          'max(ms)',read_max,freeze.max(),'stock',result['terminal_stock'],'score',score,flush=True)
    return result

if __name__=='__main__':
    verify(Path(sys.argv[1]),(Path(sys.argv[1])/'ui-verified.json').exists())
