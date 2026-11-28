"""Seal a native TH08 four-tick trajectory and audit two prospective reserve potentials."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.audit_th08_live_tick4 import game_word, read_records
from scripts.original_camera_heads import policy_heads
from scripts.run_th08_trial_camera import mask_for
from scripts.th08_camera_history import reconstruct
from scripts.qualify_th08_trial_credit import changes


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,content):
    temp=path.with_suffix(".pending")
    temp.write_text(json.dumps(content,indent=2)+"\n",encoding="utf-8")
    temp.replace(path)


def seal(raw,model,rank_template,out):
    out.mkdir(exist_ok=True,parents=True)
    receipt=json.loads((raw/"receipt.json").read_bytes())
    times=[json.loads(line) for line in (raw/"timing.jsonl").read_text().splitlines()]
    idx=np.array([i for i,r in enumerate(times) if r["phase"]=="policy"])
    play=[times[i] for i in idx];n=len(play)
    assert receipt["outcome"]=="original_continue_prompt" and receipt["requested_rank"]=="Normal" and n>4
    assert receipt["policy_sha256"]==digest(model)
    assert receipt["terminal_sha256"]==digest(raw/"terminal.png")
    assert times[-1]["phase"]=="terminal" and sum(r["phase"]=="terminal" for r in times)==1
    assert (raw/"build.sha256").read_text().split()[0]=="3a6b4a0de24f916e669cf642ffd4f7c22ada1d319d9ce8ca8494fe96a2f5f5c1"
    notes=[]
    if list(idx)!=list(range(idx[0],idx[-1]+1)):notes.append("Uncontrolled or clock-unavailable interval inside the policy trajectory")
    with np.load(raw/"behavior.npz",allow_pickle=False) as z:arrays={k:z[k].copy() for k in ("frames","action","logp","value")}
    assert arrays["frames"].shape==(n,112,96) and arrays["action"].shape==arrays["logp"].shape==(n,4)
    physical=np.array([r["mask"] for r in play]);assert np.array_equal(physical,[mask_for(a) for a in arrays["action"]])
    torch.set_num_threads(1)
    actor=torch.jit.load(str(model),map_location="cpu").eval();logp_error=critic_error=0.
    with torch.inference_mode():
        for i in range(0,n,48):
            hi=min(n,i+48);x=torch.from_numpy(reconstruct(arrays["frames"],i,hi,1)).float()/255.
            heads,value=policy_heads(actor(x),len(x));a=torch.as_tensor(arrays["action"][i:hi],dtype=torch.int64)
            lp=torch.stack([head.log_softmax(1).gather(1,a[:,k:k+1])[:,0] for k,head in enumerate(heads)],dim=1)
            logp_error=max(logp_error,float(np.abs(lp.numpy()-arrays["logp"][i:hi]).max()))
            critic_error=max(critic_error,float(np.abs(value.flatten().numpy()-arrays["value"][i:hi]).max()))
    if logp_error>1e-5 or critic_error>1e-5:notes.append("Independent behavior forward failed")
    with np.load(raw/"native-observers.npz",allow_pickle=False) as obs:
        hud,prompt=obs["hud"],obs["prompt"][-1]
    refbase=Path(__file__).resolve().parents[1]/"tools/rollout-worker-probe/th08-trial"
    import cv2
    with np.load(refbase/"continue-edges.npz") as z:
        m=z["mask"];bright=prompt[3:37,2:215].astype(np.float32)
        original=z["edge"][m];sample=(bright-cv2.GaussianBlur(bright,(0,0),1.35))[m]
        correlation=float(sample@original/(np.linalg.norm(sample)*np.linalg.norm(original)))
    assert correlation>=.78 and abs(correlation-times[-1]["continue_structure"])<1e-4
    glyph=hud[idx,170:203,104:205].astype(np.int16);r,g,b=np.moveaxis(glyph,-1,0)
    blue=(b>85)&(b-r>35)&(b-g>18);ref=np.load(rank_template)
    normal=(blue&ref).sum((1,2))/np.maximum((blue|ref).sum((1,2)),1)
    if np.mean(normal>.70)<.97:notes.append("Normal HUD rank coverage insufficient")
    for y,key in ((74,"spares"),(90,"bombs")):
        cells=[]
        for x in range(66,179,16):
            rr,gg,bb=np.moveaxis(hud[:,y-6:y+7,x-6:x+7].astype(np.int16),-1,0)
            active=((rr>125)&(rr-gg>32)&(rr-bb>36)) if key=="spares" else ((bb>110)&(bb-rr>20)&(bb-gg>28))
            cells.append(active.sum((1,2))>=16)
        for row,occupied in zip(times,np.array(cells).T):
            count=int(occupied.sum()); stock=count if occupied[:count].all() else None
            assert stock==row[key]
    assert [play[0]["spares"],play[0]["bombs"]]==[2,3] and times[-1]["spares"]==0
    assert all(row["spares"] is not None and row["bombs"] is not None for row in play)
    spares=np.array([row["spares"] for row in play],np.int16)
    bombs=np.array([row["bombs"] for row in play],np.int16)
    next_spares=np.r_[spares[1:],np.int16(times[-1]["spares"])]
    next_bombs=np.r_[bombs[1:],np.int16(times[-1]["bombs"])]
    gates=np.genfromtxt(raw/"gates.tsv",names=True,dtype=None,encoding="ascii",ndmin=1)
    assert len(gates)==n and np.array_equal(gates["index"],np.arange(n)) and np.array_equal(gates["x11_mask"],physical)
    stages=np.int16(gates["stage"]);target=np.int32(gates["apply_target"]+1);words=read_records(raw)
    trace=np.genfromtxt(raw/"clock.tsv",names=True,dtype=None,encoding="ascii",ndmin=1)
    valid=(trace["stable"]==1)&(trace["mode"]==0)&(trace["flags"]==4)&(trace["buffer_read"]==1)&(trace["origin"]!=0)
    valid &= (trace["cursor"]==trace["origin"]+36+2*trace["tick"]) & (trace["saved_current"]==trace["word"])
    valid &= (trace["after_ns"]>=play[0]["monotonic_ns"])&(trace["before_ns"]<=times[-1]["monotonic_ns"])
    end_stage=int(stages[-1])
    known=valid&(trace["stage_index"]==end_stage-1)
    terminal_rows=(trace["stable"]==1)&(trace["mode"]==0)&(trace["flags"]==0)&(trace["stage_index"]==end_stage-1)
    terminal_rows &= (trace["before_ns"]>=np.max(trace["after_ns"][known]))&(trace["before_ns"]<=times[-1]["monotonic_ns"])
    terminal_ticks=np.unique(trace["tick"][terminal_rows])
    if len(terminal_ticks)!=1:notes.append("Unique native stopped-stage tick missing for the original terminal")
    final_tick=int(terminal_ticks[0]) if len(terminal_ticks)==1 else int(np.max(trace["tick"][known]))
    end_tick=min(final_tick+1,len(words[end_stage]))
    if end_tick!=final_tick+1:notes.append("Native recording lacks the through-terminal key")
    physical_requests=n
    terminal_proposal=None
    # A command can reach X11 while the game has already consumed the final tick.
    # Preserve the entire physical recording and every proposal's re-forward above.
    # Only the unique last command can be removed from the optimizer view.
    if n>1 and int(target[-1])>final_tick and len(terminal_ticks)==1:
        last,previous=gates[-1],gates[-2]
        prior_word=game_word(int(physical[-2]));proposed_word=game_word(int(physical[-1]))
        raw_received=terminal_rows & (trace["before_ns"]>=last["x_ack_ns"]) & (trace["raw"]==proposed_word) & (trace["word"]==prior_word)
        prior_slots=words[end_stage][int(target[-2]):end_tick]
        earlier_live=valid & (trace["tick"]==final_tick) & (trace["word"]==prior_word)
        eligible_after=int(last["apply_observed"])+1
        proven=(stages[-1]==stages[-2] and prior_word!=proposed_word and eligible_after>final_tick
                and end_tick==final_tick+1 and len(prior_slots)>0 and np.all(prior_slots==prior_word)
                and np.any(raw_received) and np.any(earlier_live))
        if proven:
            terminal_proposal=dict(raw_index=n-1,actual_x11_mask=int(physical[-1]),requested_game_word=proposed_word,
                x11_ack_ns=int(last["x_ack_ns"]),native_application_tick=int(last["apply_observed"]),
                nominal_first_consumption_tick=int(target[-1]),earliest_after_application=eligible_after,
                terminal_tick=final_tick,native_exposure=0,preceding_index=n-2,preceding_word=prior_word,
                preceding_native_slots=list(map(int,prior_slots)),source_raw_only_samples=int(raw_received.sum()),
                preserved_policy_action=arrays["action"][-1].tolist(),preserved_logp=arrays["logp"][-1].tolist(),
                preserved_value=float(arrays["value"][-1]),raw_behavior_sha256=digest(raw/"behavior.npz"))
            arrays={k:v[:-1] for k,v in arrays.items()}
            n-=1;idx=idx[:-1];play=play[:-1];gates=gates[:-1];stages=stages[:-1];target=target[:-1]
            physical=physical[:-1];spares=spares[:-1];bombs=bombs[:-1]
            next_spares=np.r_[spares[1:],np.int16(times[-1]["spares"])]
            next_bombs=np.r_[bombs[1:],np.int16(times[-1]["bombs"])]
        else:notes.append("Post-terminal proposal cannot be attributed unambiguously to the preceding consumed action")
    dest_tick=np.r_[target[1:],np.int32(final_tick)]
    dest_stage=np.r_[stages[1:],np.int16(end_stage)]
    if len(set(stages))!=1:notes.append("Cross-stage progress trace requires independent boundary qualification")
    match=np.zeros(n,bool);any_match=np.zeros(n,bool);proven_edge=np.zeros(n,bool)
    diff=np.zeros(n,bool);exposure=np.zeros(n,np.int16);word_actual=np.zeros((n,4),np.uint16)
    first_native=np.full(n,-1,np.int32)
    for i,(stage,tick) in enumerate(zip(stages,target)):
        table=words[int(stage)]
        end=int(target[i+1]) if i+1<n and stages[i+1]==stage else (end_tick if i==n-1 else len(table))
        end=min(end,len(table));begin=int(tick)
        if end<=begin or begin<1:continue
        slots=table[begin:end];w=game_word(int(physical[i]));good=np.flatnonzero(slots==w)
        exposure[i]=len(slots)
        if len(good):any_match[i]=True;first_native[i]=int(begin+good[0])
        match[i]=np.all(slots==w)
        word_actual[i,:min(4,len(slots))]=slots[:4]
        diff[i]=(i==0 or physical[i]!=physical[i-1])
        proven_edge[i]=diff[i] and table[begin-1]!=w and table[begin]==w
    intervals_exact=(np.diff(target)==4)&(np.diff(stages)==0)&match[:-1]
    nominal_gates=(gates["apply_target"]==gates["apply_observed"])
    exact_fraction=float(intervals_exact.mean())
    if exact_fraction<.99:notes.append("Fewer than 99% of intra-credit native four-tick intervals match throughout")
    if any_match.sum()!=n:notes.append("At least one sampled action has no native consumption in its assigned interval")
    if nominal_gates.mean()<.99:notes.append("Fewer than 99% of X11 gates were observed on the scheduled tick")
    for s in set(stages):
        v=valid&(trace["stage_index"]==s-1)&(trace["tick"]<len(words[int(s)]))
        if np.any(words[int(s)][trace["tick"][v]]!=trace["word"][v]):notes.append("Persisted record disagrees with independent concurrent observer")
    loss_events=changes(list(spares)+[times[-1]["spares"]],2)
    if sum(d for i,d in loss_events)!=-2 or spares[-1]!=0:notes.append("Stock ledger failed to reconcile")
    loss=(spares-next_spares).astype(np.int8)
    loss[-1]+=1
    done=np.zeros(n,bool);done[-1]=True
    if loss.sum()!=3:notes.append("Final active life ledger mismatch")
    bomb_spent=np.zeros(n,np.int8)
    for i,d in changes(list(bombs)+[times[-1]["bombs"]],3):
        if d<0 and all(abs(j-i)>6 for j,ld in loss_events if ld<0):
            bomb_spent[i]=-d
    # Stage progress and reserve differences are distinct quantities.
    progress=(dest_tick-target).astype(np.float64)/15652.
    phi=spares.astype(np.float64)+1;next_phi=next_spares.astype(np.float64)+1;next_phi[-1]=0
    delta=.999*next_phi-phi
    disc=np.power(.999,np.arange(n,dtype=np.float64))
    telescope=float(np.dot(disc,delta))
    assert abs(telescope+3)<1e-8
    arrays.update(native_stage=stages,native_tick=target,native_first_matching_tick=first_native,native_next_tick=dest_tick,
        native_next_stage=dest_stage,native_word_slots=word_actual,native_ticks_under_command=exposure,x11_target_tick=np.int32(gates["apply_target"]),
        x11_observed_tick=np.int32(gates["apply_observed"]),spares=spares,bombs=bombs,next_spares=next_spares,next_bombs=next_bombs,
        life_loss=loss,bomb_spent=bomb_spent,normalized_stage1_progress=np.float32(target/15652.),
        next_normalized_stage1_progress=np.float32(dest_tick/15652.),stage_progress=np.float32(progress),
        reserve_phi=np.float32(phi),reserve_next_phi=np.float32(next_phi),reserve_potential_delta=np.float32(delta),terminated=done)
    output=out/f'th08-{receipt["seed"]}.npz'
    np.savez_compressed(output,**arrays)
    stats=dict(seed=receipt["seed"],role=receipt["role"],policy_sha256=digest(model),source_receipt_sha256=digest(raw/"receipt.json"),
        file=output.name,sha256=digest(output),game_sha256=(raw/"build.sha256").read_text().split()[0],
        stage_clock_trace_sha256=digest(raw/"clock.tsv"),timing_sha256=digest(raw/"timing.jsonl"),input_gates_sha256=digest(raw/"gates.tsv"),
        stock_trace_sha256=digest(raw/"native-observers.npz"),terminal_photo_sha256=digest(raw/"terminal.png"),outcome="game_over",continues=0,
        decisions=n,physical_sampled_requests=physical_requests,all_physical_requests_reforwarded=physical_requests,
        excluded_terminal_proposal=terminal_proposal,raw_behavior_sha256=digest(raw/"behavior.npz"),
        action_masks_with_native_consumption=int(any_match.sum()),exact_complete_intervals=int(intervals_exact.sum()),
        native_intervals=n-1,exact_four_tick_fraction=exact_fraction,exact_x11_gates=int(nominal_gates.sum()),native_edges_verifiable=int(diff.sum()),
        exactly_observed_native_edges=int(proven_edge.sum()),no_input_transition_frames_after_final_decision=len(times)-2-idx[-1].item(),
        first_tick=int(target[0]),first_stage=int(stages[0]),terminal_tick=final_tick,terminal_stage=end_stage,terminal_record_exclusive=end_tick,
        active_native_ticks_in_last_action=int(exposure[-1]),first_stock=[int(spares[0]),int(bombs[0])],
        final_stock=[int(spares[-1]),int(bombs[-1])],terminal_stock=[times[-1]["spares"],times[-1]["bombs"]],
        life_events=[dict(decision=int(i),stage=int(stages[i]),native_tick=int(target[i]),next_tick=int(dest_tick[i]),
            spares=int(spares[i]),next_spares=int(next_spares[i]),bombs=int(bombs[i]),next_bombs=int(next_bombs[i]),
            event="original_game_over" if i==n-1 else "spare_lost") for i in np.flatnonzero(loss)],
        max_logp_error=logp_error,max_critic_error=critic_error,normal_rank_samples=int(sum(normal[:n]>.70)),
        physical_normal_rank_samples=int(sum(normal>.70)),terminal_glyph=correlation,
        reward_audit=dict(stage1_denominator=15652,progress_start=float(target[0]/15652),progress_terminal=final_tick/15652.,
            paid_progress=float(progress.sum()),positive_same_life_bomb_uses=int(bomb_spent.sum()),
            gamma=.999,lambda_life_0_discounted_delta=0.,lambda_life_1_discounted_delta=telescope,
            lambda_life_1_undiscounted_delta=float(delta.sum()),first_potential=3.,final_absorbing_potential=0.),
        qualified=not notes,issues=notes)
    save(out/f'{receipt["seed"]}.json',stats)
    return stats


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("raw","model","rank-template","out"):p.add_argument("--"+arg,type=Path,required=True)
    a=p.parse_args();print(json.dumps(seal(a.raw,a.model,a.rank_template,a.out),indent=2))
