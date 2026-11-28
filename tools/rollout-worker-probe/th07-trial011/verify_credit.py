"""Check consumed original keys for every exact decision window; never silently discard a middle action."""
import argparse
import csv
import hashlib
import json
import struct
import sys
from pathlib import Path
import numpy as np
from PIL import Image


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def replay_stages(path, decoder_root):
    sys.path.insert(0,str(decoder_root))
    from inspect_rpy import decompress
    raw=bytearray(path.read_bytes())
    assert raw[:6].hex()=="543752501110"
    raw[16:]=bytes((v-raw[13]-7*i)%256 for i,v in enumerate(raw[16:]))
    compressed,expanded,*offsets=struct.unpack_from("<9I",raw,20)
    assert compressed==len(raw)-84 and 0<expanded<3_000_000
    payload=decompress(bytes(raw[84:]),expanded)
    assert list(payload[2:4])==[0,1]
    stages={}
    for stage,off in enumerate(offsets,1):
        if not off:continue
        # The second seven-offset table points to optional per-stage slowdown samples.
        # Its byte stream follows the original input words and is not action data.
        slowdown=struct.unpack_from("<7I",raw,56)
        next_offset=min(x for x in [*offsets,*slowdown,len(payload)+84] if x>off)
        buf=payload[off-84+40:next_offset-84]
        assert len(buf)%4==0
        stages[stage]=np.frombuffer(buf,"<u2").reshape(-1,2)[:,0]
    return stages,10*struct.unpack_from("<I",payload,24)[0]


def color_mask(path,rect):
    return np.asarray(Image.open(path).convert("L").crop(rect))>180


def verify(folder,base,study):
    cfg=json.loads((study/"protocol.json").read_text())
    receipt=json.loads((folder/"receipt.pending-visual.json").read_text())
    arm=next(a for a,sha in cfg["models"].items() if sha==receipt["policy_sha256"])
    stages,score=replay_stages(folder/"official-policy.rpy",base/"online")
    controls=np.genfromtxt(folder/"controls.csv",delimiter=",",names=True,dtype=None)
    ap=np.genfromtxt(folder/"applied.csv",delimiter=",",names=True,dtype=None)
    n=np.genfromtxt(folder/"native-input.csv",delimiter=",",names=True,dtype=None)
    with np.load(folder/"policy-trace.npz",allow_pickle=False)as z:
        a=z["action"];t=z["native_tick"];frames=z["frames"]
    assert len(a)==len(t)==len(ap)==len(controls)==len(frames)==receipt["actions"]
    assert np.array_equal(t,ap["camera_tick"]) and np.array_equal(t,controls["native_tick"])
    assert np.array_equal(np.arange(len(t)),ap["sequence"])
    masks=np.array([16,0,32])[a[:,0]//3]|np.array([64,0,128])[a[:,0]%3]|(4*a[:,1])|a[:,2]|(2*a[:,3])
    assert np.array_equal(masks,ap["requested_word"])
    data,terminal=[],{}
    g=np.r_[1,1+np.cumsum(np.diff(t)<0)]
    asg=1+np.r_[0,np.cumsum(np.diff(n["tick"])<-32)]
    stage1_max=int(np.max(n["tick"][asg==1]))
    obs_keys={}
    for (_,nt,word,prev),stage in zip(n,asg):
        obs_keys.setdefault((int(stage),int(nt)),set()).add(int(word)&247)
    # During original recording the saved source slot is one later than the current
    # native live-input tick. Replay playback itself uses a different indexing phase.
    # Verify the frozen offset against every final observed word, independent of any policy.
    final_native={}
    for row,st in zip(n,asg):final_native[(int(st),int(row["tick"]))]=int(row["word"])&247
    offset_trials={}
    for delta in (-2,-1,0,1,2):
        checks=[v==(int(stages[st][tick+delta])&247) for (st,tick),v in final_native.items() if st in stages and 0<=tick+delta<len(stages[st])]
        offset_trials[delta]={"same":sum(checks),"count":len(checks),"rate":sum(checks)/max(1,len(checks))}
    recording_alignment_ok=offset_trials[1]["rate"]>=.995 and offset_trials[1]["rate"]>max(offset_trials[k]["rate"] for k in (-2,-1,0,2))
    stale=[];not_exact=[];partial=[];skipped=0;success=0;total=0;direct=0
    for i,(stage,nt,mask) in enumerate(zip(g,t,masks)):
        st=int(stage);nt=int(nt);want=int(mask)
        if st not in stages:raise ValueError(f"Unknown stage {st}")
        ticks=np.arange(nt+1,nt+5);src=stages[st]
        last_physical=int(n["tick"][asg==st].max())
        available=ticks[ticks<=last_physical]
        if len(available)<4:
            assert i==len(t)-1 or g[i+1]!=stage, "Partial middle window"
            assert np.all((src[available+1]&247)==want) if len(available) else True
            partial.append(dict(i=i,tick=nt,available_verified=len(available)));continue
        if ticks[-1]+1>=len(src):
            if i<len(t)-1 and g[i+1]==stage:
                raise ValueError("Nonterminal decision outside original replay")
            partial.append(i);continue
        live=[obs_keys.get((st,int(k)),set())for k in ticks]
        source=src[ticks+1]&247
        any_match=bool(np.any(source==want))
        independently_observed=any(want in words for words in live)
        direct+=int(independently_observed)
        if not any_match or not independently_observed:stale.append(i)
        gap=(i+1<len(t)and g[i+1]==stage and t[i+1]!=nt+4)
        if gap:skipped+=int((t[i+1]-nt)//4)-1
        exact=bool(np.all(source==want))and not gap
        total+=1;success+=int(exact)
        if not exact:
            not_exact.append(dict(i=i,tick=nt,requested=want,original=source.tolist(),gap=bool(gap)))
    reference=base/"online/held-six-ppo-2080751/settled-scorecard.png"
    scorecard=folder/"settled-scorecard.png"
    def match(rect):
        r=color_mask(reference,rect);b=color_mask(scorecard,rect)
        return float((r&b).sum()/max(1,(r|b).sum()))
    zero_continue=match((380,190,405,212))>.94
    three_misses=match((380,212,405,234))>.94
    score_rank_normal=match((311,146,402,170))>.94
    initial=[int(controls[k][0]) for k in ("spares","bombs")]
    manual_path=study/"tool/reference-scorecards.json"
    originals=json.loads(manual_path.read_text())
    calibrated=[]
    for old,labels in originals.items():
        old_path=base/"online"/old/"settled-scorecard.png"
        if old_path.exists():
            rect=(360,234,404,256)
            u=color_mask(old_path,rect);v=color_mask(scorecard,rect)
            calibrated.append((float((u&v).sum()/max(1,(u|v).sum())),labels["bombs"]))
    calibrated.sort(reverse=True)
    bomb_count=calibrated[0][1] if calibrated and calibrated[0][0]>=.96 else None
    active_ns=(ap["resume_ns"]-ap["freeze_start_ns"])/1e6 if "resume_ns" in ap.dtype.names else None
    real_duration=(int(n["ns"][-1])-int(n["ns"][0]))/1e9
    losses=[];before=2
    for i in range(1,len(t)-2):
        if before>0 and np.all(controls["spares"][i:i+3]==before-1):
            losses.append(dict(stage=int(g[i]),tick=int(t[i]),spares_before=before,bombs_prior_hud=int(controls["bombs"][i-1])))
            before-=1
    # The last fatal event is bracketed by the final live decision and genuine Continue.
    losses.append(dict(stage=int(g[-1]),tick_upper=int(np.max(n["tick"][asg==g[-1]])),spares_before=0,bombs_prior_hud=int(controls["bombs"][-1])))
    ratio=success/(total+skipped)
    passed=bool(recording_alignment_ok and ratio>=.995 and not stale and initial==[2,3] and zero_continue and three_misses and score_rank_normal and before==0)
    report=dict(role="engineering"if folder.name.startswith("engineering")else"held",seed=receipt["seed"],arm=arm,policy_sha256=receipt["policy_sha256"],
                opening=initial,confirmed_zero_continue=zero_continue,scorecard_three_misses=three_misses,normal_rank_scorecard=score_rank_normal,
                stage2=2 in stages,stage1_native_ticks=stage1_max,stage1_deaths=sum(x["stage"]==1 for x in losses),losses=losses,
                stock_at_final_camera=[int(controls[k][-1])for k in ("spares","bombs")],game_score=score,
                stages={s:len(w) for s,w in stages.items()},decisions=len(t),native_observed_rows=len(n),strict_exact=success,
                complete_windows=total,skipped_decisions=skipped,exact_fraction=ratio,any_native_consumption=direct,stale_indices=stale,
                incomplete_terminal_decisions=partial,nonexact_windows=not_exact,passed=passed,
                ms_per_decision=receipt["median_step_ms"],original_bombs=bomb_count,
                recording_offset_comparison=offset_trials, correct_native_tick_fraction=float(np.mean(t==ap["applied_tick"])),elapsed_active_real_seconds=real_duration,
                native_stage1_ticks_per_active_wall_second=stage1_max/real_duration if len(stages)==1 else None,
                freeze_ms_p50_p95_p99_max=np.percentile(active_ns,[50,95,99,100]).tolist() if active_ns is not None else None,
                protocol_sha256=digest(study/"protocol.json"),
                evidence={name:digest(folder/name) for name in ("native-input.csv","applied.csv","controls.csv","policy-trace.npz","official-policy.rpy","settled-scorecard.png","settled-continue.png","full-stock-opening.png")})
    (folder/"verified.json").write_text(json.dumps(report,indent=2)+"\n")
    print(folder.name,"passed",passed,"stage",tuple(stages),"valid",round(ratio,6),"missing",stale[:5],"bad",not_exact[:2],"inventory",initial,zero_continue,three_misses,score_rank_normal,flush=True)
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("folder",type=Path);p.add_argument("--base",type=Path,required=True);p.add_argument("--study",type=Path,required=True)
    args=p.parse_args();verify(args.folder,args.base,args.study)
