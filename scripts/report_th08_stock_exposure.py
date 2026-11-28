"""Extract native exposure and pre-loss Bomb stock for one sealed physical TH08 credit."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.audit_th08_live_tick4 import game_word, read_records, sha


def summarize(raw,packed,receipt):
    r=json.loads(receipt.read_bytes())
    if not r["qualified"]:raise ValueError("Strict stock metrics require a qualified original")
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
    return dict(seed=r["seed"],phase=r["role"],native_inputs=int(exposure.sum()),stage_spans=spans,
        native_ticks_matching_requested_mask=matching,native_ticks_stale_or_different=int(exposure.sum())-matching,
        native_record_sha256={p.name:sha(p) for p in sorted(raw.glob("native-record*.bin"))},
        terminal_photo_sha256=r["terminal_photo_sha256"],receipt_sha256=sha(receipt),packed_sha256=sha(packed),
        final_native_stage=r["terminal_stage"],final_native_tick=r["terminal_tick"],original_outcome=r["outcome"],
        first_stock=r["first_stock"],last_playable_stock=r["final_stock"],terminal_stock=r["terminal_stock"],
        first_loss_native_interval=[events[0]["native_tick"],events[0]["next_tick"]] if events else None,
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
