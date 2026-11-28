"""Check full native cadence, observer latency, archive lifecycle and full color movie for one engineering original."""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from scripts.audit_th08_live_tick4 import read_records,sha
from scripts.report_th08_stock_exposure import terminal_stage_evidence


def latency(values):
    if not len(values):return None
    return dict(samples=len(values),median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),
                p999_ms=float(np.percentile(values,99.9)),max_ms=float(np.max(values)),
                over_30ms=int((values>=30).sum()),over_100ms=int((values>100).sum()))


def read_tsv(path):
    with path.open() as stream:
        rows=list(csv.DictReader(stream,delimiter="\t"))
    if len(rows)!=1:raise ValueError(f"Expected one seal acknowledgement in {path}")
    return {k:int(v) for k,v in rows[0].items()}


def audit(raw,receipt):
    times=[json.loads(s) for s in (raw/"timing.jsonl").read_text().splitlines()]
    credit=json.loads(receipt.read_bytes())
    if credit["role"]!="engineering-zero-shot-original" or times[-1]["phase"]!="terminal":
        raise ValueError("Engineering original terminal missing")
    start=next(r["monotonic_ns"] for r in times if r["phase"]=="policy")
    end=times[-1]["monotonic_ns"]
    t=np.genfromtxt(raw/"clock.tsv",names=True,dtype=None,encoding="ascii")
    elapsed=(t["after_ns"]-t["before_ns"])/1e6
    during=(t["after_ns"]>=start)&(t["before_ns"]<=end)
    at_boundary=np.zeros(len(t),dtype=bool)
    previous={}
    for j,r in enumerate(t):
        s=int(r["stage_index"]);tick=int(r["tick"]);f=int(r["flags"]);n=tick+int(f==0)
        if r["stable"] and r["mode"]==0 and s<3 and r["origin"] and r["cursor"]==r["origin"]+36+2*tick and n<65535:
            old=previous.get(s,0)
            if (f==4 and n>=old+32) or (f==0 and n>old):
                at_boundary[j]=True;previous[s]=n
    movie=raw/"original-observations.mp4"
    player=cv2.VideoCapture(str(movie));frames=0
    while True:
        ok,frame=player.read()
        if not ok:break
        if frame.shape!=(480,640,3):raise ValueError("Movie frame has wrong camera dimensions")
        frames+=1
    player.release()
    recording=json.loads((raw/"movie-pipeline.json").read_bytes())
    source=json.loads((raw/"receipt.json").read_bytes())
    native=read_records(raw)
    writer=read_tsv(raw/"native-archive-sealed.tsv")
    handoff=read_tsv(raw/"native-archive-bracket-status.tsv")
    counts=[len(native.get(s,())) for s in (1,2,3)]
    interval=latency(elapsed[during])
    duration=(end-start)/1e9
    endpoint=terminal_stage_evidence(raw,credit)
    physical=dict(all_masks=credit["action_masks_with_native_consumption"]==credit["decisions"],
        strict_native=credit["qualified"],exact_four=credit["exact_four_tick_fraction"]>=.998,
        observer_tail=interval["p999_ms"]<30 and interval["max_ms"]<=100,
        archive=(counts==[writer[f"stage{s}"] for s in (1,2,3)]==[handoff[f"stage{s}_words"] for s in (1,2,3)]
                 and handoff["reader_exit"]==0 and handoff["ack_ns"]>end and counts[credit["terminal_stage"]-1]==credit["terminal_tick"]+1),
        color_video=frames==len(times)==recording["encoded_frames"] and recording["backpressure_ns"]==0 and sha(movie)==source["movie_sha256"],
        stock_terminal=endpoint["stage"]==credit["terminal_stage"] and credit["first_stock"]==[2,3] and credit["continues"]==0)
    worst=np.flatnonzero(during)[np.argsort(elapsed[during])[-5:][::-1]]
    return dict(seed=credit["seed"],all_gates_passed=all(physical.values()),gates=physical,actor_sha256=credit["policy_sha256"],
        observer_gameplay=interval,observer_all=latency(elapsed),first_query_rtt_ms=float(elapsed[0]),
        observer_snapshot_boundary=latency(elapsed[during&at_boundary]),observer_off_boundary=latency(elapsed[during&~at_boundary]),
        snapshot_groups="Based on eligible stable stage/cursor and advance of at least 32 active ticks or a new stopped tick. All physical q rows, including invalid rows, stay in the primary latency denominator.",
        worst_queries=[dict(row=int(j),stage_index=int(t["stage_index"][j]),native_tick=int(t["tick"][j]),rtt_ms=float(elapsed[j]),
            between_polls_ms=float((t["before_ns"][j+1]-t["after_ns"][j])/1e6) if j+1<len(t) else None,snapshot_boundary=bool(at_boundary[j])) for j in worst],
        exact_intervals=credit["exact_complete_intervals"],total_intervals=credit["native_intervals"],exact_fraction=credit["exact_four_tick_fraction"],
        all_consumed_retained=credit["action_masks_with_native_consumption"],decisions=credit["decisions"],raw_requested=credit["physical_sampled_requests"],
        native_tick_at_game_over=credit["terminal_tick"],final_stock=credit["terminal_stock"],terminal=endpoint,
        movie_sha256=sha(movie),full_color_frames=frames,timing_frames=len(times),movie_writer=recording,seal=writer,bracket=handoff,
        first_action_to_terminal_seconds=duration,native_fps=(credit["terminal_tick"]-credit["first_tick"]+1)/duration,
        native_record_sha256={p.name:sha(p) for p in raw.glob("native-record-*.bin")},
        native_capture_sha256=sha(raw/"clock.tsv"),behavior_sha256=source["behavior_sha256"],credit_sha256=sha(receipt))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("raw","receipt","out"):p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();result=audit(a.raw,a.receipt)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
