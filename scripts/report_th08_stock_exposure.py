"""Extract native exposure and pre-loss Bomb stock for one sealed physical TH08 credit."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.audit_th08_live_tick4 import game_word, read_records, sha


def terminal_stage_evidence(raw,receipt):
    from scripts.score_th08_original_ranking import inspect_results
    timing=[json.loads(line) for line in (raw/"timing.jsonl").read_text().splitlines()]
    start=next(row["monotonic_ns"] for row in timing if row["phase"]=="policy")
    end=timing[-1]["monotonic_ns"]
    if timing[-1]["phase"]!="terminal" or receipt["terminal_photo_sha256"]!=sha(raw/"terminal.png"):
        raise ValueError("Missing original Game Over evidence")
    trace=np.genfromtxt(raw/"clock.tsv",names=True,dtype=None,encoding="ascii",ndmin=1)
    interval=(trace["stable"]==1)&(trace["mode"]==0)&(trace["before_ns"]>=start)&(trace["after_ns"]<=end)
    live=interval&(trace["flags"]==4)&(trace["origin"]!=0)&(trace["buffer_read"]==1)
    live&=(trace["cursor"]==trace["origin"]+36+2*trace["tick"])&(trace["saved_current"]==trace["word"])
    rank=inspect_results(raw)
    stages=sorted(set(int(s)+1 for s in trace["stage_index"][live]))
    recorded=read_records(raw)
    verified={}
    for stage in stages:
        seq=recorded.get(stage)
        if seq is None:continue
        rows=trace[live&(trace["stage_index"]==stage-1)]
        valid=rows[(rows["tick"]>=0)&(rows["tick"]<len(seq))]
        if len(valid)>0 and np.all(seq[valid["tick"]]==valid["word"]):
            verified[stage]=len(set(valid["tick"]))
    native=None
    if stages and all(s in verified for s in stages) and stages==list(range(1,stages[-1]+1)):
        stopped=trace[interval&(trace["flags"]==0)&(trace["before_ns"]>=trace["after_ns"][live].max())]
        pairs={(int(s)+1,int(t)) for s,t in zip(stopped["stage_index"],stopped["tick"])}
        expected=(int(receipt["terminal_stage"]),int(receipt["terminal_tick"]))
        if pairs=={expected} and stages[-1]==expected[0]:native=stages[-1]
    score=rank["stage"]
    agreed=(native if native is not None else score) if score is None or native in (None,score) else None
    return dict(stage=agreed,stage2=agreed>=2 if agreed is not None else None,original_ranking=rank,
        native_terminal_stage=native,distinct_matching_native_ticks_by_stage=verified,original_photo_sha256=receipt["terminal_photo_sha256"],
        rank_photos_sha256={o["file"]:sha(raw/o["file"]) for o in rank["observations"]},clock_sha256=sha(raw/"clock.tsv"))


def postprompt_stage1_tick(raw,receipt):
    """Qualify a frozen original tick in the compulsory no-menu-input interval after Game Over."""
    timing=[json.loads(s) for s in (raw/"timing.jsonl").read_text().splitlines()]
    if timing[-1]["phase"]!="terminal" or sha(raw/"terminal.png")!=receipt["terminal_photo_sha256"]:
        raise ValueError("Terminal source unavailable")
    end=timing[-1]["monotonic_ns"]
    trace=np.genfromtxt(raw/"clock.tsv",names=True,dtype=None,encoding="ascii",ndmin=1)
    replay=read_records(raw)[1]
    selected=trace[(trace["before_ns"]>=end)&(trace["after_ns"]<=end+1_500_000_000)
        &(trace["mode"]==0)&(trace["stable"]==1)&(trace["stage_index"]==0)&(trace["origin"]!=0)
        &(trace["cursor"]==trace["origin"]+36+2*trace["tick"])
        &(trace["buffer_read"]==1)&(trace["saved_current"]==trace["word"])]
    ticks=np.unique(selected["tick"])
    ok=(len(ticks)==1 and len(selected)>=2 and (selected["flags"]==0).all() and
        int(ticks[0])+1==len(replay) and (selected["word"]==replay[-1]).all())
    return dict(verified=bool(ok),first_native_stopped_tick=int(ticks[0]) if ok else None,
        stopped_rows=int(len(selected)),candidate_ticks=[int(i) for i in ticks],
        earliest_lower_bound_after_prompt_seconds=(int(selected["before_ns"].min())-end)/1e9 if len(selected) else None,
        latest_upper_bound_after_prompt_seconds=(int(selected["after_ns"].max())-end)/1e9 if len(selected) else None,
        original_words=int(len(replay)),terminal_photo_sha256=receipt["terminal_photo_sha256"],clock_sha256=sha(raw/"clock.tsv"),
        protocol="Same deployed postcredit-ranking.sh releases all controls and waits at least 1.7 s after the original prompt before first menu input. Only stable matching native rows up to +1.5 s are used.")


def summarize(raw,packed,receipt,*,require_strict=True):
    r=json.loads(receipt.read_bytes())
    if require_strict and not r["qualified"]:raise ValueError("Strict stock metrics require a qualified original")
    if sha(packed)!=r["sha256"] or sha(raw/"gates.tsv")!=r["input_gates_sha256"]:
        raise ValueError("Credit data do not match signed receipt")
    with np.load(packed,allow_pickle=False) as z:
        ticks=z["native_tick"];stages=z["native_stage"];exposure=z["native_ticks_under_command"]
        lives=z["life_loss"];before=z["bombs"];next_bombs=z["next_bombs"];spent=z["bomb_spent"]
    with np.load(raw/"behavior.npz") as z:raw_action=z["action"]
    started=np.r_[True,np.diff(raw_action[:,3])!=0]&(raw_action[:,3]>0)
    by_stage=read_records(raw)
    issued=np.genfromtxt(raw/"gates.tsv",names=True,dtype=None,encoding="ascii",ndmin=1)
    if len(issued)!=len(raw_action) or len(ticks)!=r["decisions"]:
        raise ValueError("Physical or retained control counts disagree")
    matching=0
    for stage,lo,count,row in zip(stages,ticks,exposure,issued):
        slots=by_stage[int(stage)][int(lo):int(lo+count)]
        matching+=int(np.count_nonzero(slots==game_word(int(row["x11_mask"]))))
    x_ticks=x_edges=0
    spans=[]
    for stage in sorted(set(stages)):
        which=np.flatnonzero(stages==stage)
        lo=int(ticks[which[0]]);hi=int(ticks[which[-1]]+exposure[which[-1]])
        seq=by_stage[int(stage)]
        active=(seq[lo:hi]&2)!=0
        was_active=bool(seq[lo-1]&2) if lo else False
        x_ticks+=int(active.sum())
        x_edges+=int((active&~np.r_[was_active,active[:-1]]).sum())
        spans.append(dict(stage=int(stage),first_consumption_window_tick=lo,last_recorded_input_tick=hi-1,
                          inclusive_input_span_ticks=hi-lo))
    events=[]
    for i in np.flatnonzero(lives>0):
        raw_event=next(event for event in r["life_events"] if event["decision"]==int(i))
        events.append(dict(**raw_event,bomb_units_before_loss=int(before[i]),
            bomb_units_next=int(next_bombs[i]),reserve_positive_before_loss=bool(before[i]>0)))
    first_loss=events[0] if events else None
    first_loss_controlled=int(exposure[:first_loss["decision"]].sum()) if first_loss else None
    return dict(seed=r["seed"],phase=r["role"],native_inputs=int(exposure.sum()),stage_spans=spans,
        **({"control_qualified":False,"technical_issues":r["issues"]} if not r["qualified"] else {}),
        native_ticks_matching_requested_mask=matching,native_ticks_stale_or_different=int(exposure.sum())-matching,
        native_record_sha256={p.name:sha(p) for p in sorted(raw.glob("native-record*.bin"))},
        terminal_photo_sha256=r["terminal_photo_sha256"],receipt_sha256=sha(receipt),packed_sha256=sha(packed),
        final_native_stage=r["terminal_stage"],final_native_tick=r["terminal_tick"],original_outcome=r["outcome"],
        first_stock=r["first_stock"],last_playable_stock=r["final_stock"],terminal_stock=r["terminal_stock"],
        first_loss_native_interval=[events[0]["native_tick"],events[0]["next_tick"]] if events else None,
        native_ticks_elapsed_to_first_loss_interval_start=first_loss_controlled,
        pre_loss_stock_source="Last pre-action HUD sample before the observed loss transition. Native event time is bounded by the paired stage-local ticks.",
        total_lost_lives=int(lives.sum()),life_loss_events=len(events),losses_with_positive_pre_loss_bomb=sum(e["reserve_positive_before_loss"] for e in events),
        losses_with_known_pre_loss_bomb=len(events),events=events,same_life_bomb_units_spent=int(spent.sum()),
        same_life_stock_decrease_events=int(np.count_nonzero(spent)),
        sampled_x_decisions=int(raw_action[:,3].sum()),sampled_x_onsets=int(started.sum()),
        actual_recorded_x_ticks=x_ticks,actual_recorded_x_onsets=x_edges,
        physical_requests=len(raw_action),retained_decisions=r["decisions"],exact_four_tick_fraction=r["exact_four_tick_fraction"])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("raw","packed","receipt","out"):p.add_argument("--"+name,required=True,type=Path)
    a=p.parse_args();a.out.write_text(json.dumps(summarize(a.raw,a.packed,a.receipt),indent=2)+"\n",encoding="utf-8")
