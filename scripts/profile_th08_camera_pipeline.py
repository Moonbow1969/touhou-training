"""Decompose same-host native policy latency from camera capture through consumed game input."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.audit_th08_live_tick4 import game_word,read_records


def profile(raw):
    logs=[json.loads(s) for s in (raw/"timing.jsonl").read_text().splitlines()]
    play=[r for r in logs if r["phase"]=="policy"]
    gates=np.genfromtxt(raw/"gates.tsv",names=True,dtype=None,encoding="ascii",ndmin=1)
    assert len(gates)==len(play)
    values={k:np.array([r[k] for r in play],np.int64) for k in
            ("monotonic_ns","camera_read_ns","hud_ready_ns","inference_start_ns","inference_end_ns","fifo_flushed_ns")}
    values.update({k:gates[k] for k in ("command_received_ns","gate_ready_ns","x_ack_ns")})
    fields=[("capture_to_rgb_read","monotonic_ns","camera_read_ns"),("read_to_hud","camera_read_ns","hud_ready_ns"),
            ("hud_to_network","hud_ready_ns","inference_start_ns"),("network_and_sampling","inference_start_ns","inference_end_ns"),
            ("sample_to_linux_input_receive","inference_end_ns","command_received_ns"),
            ("capture_to_linux_input_receive","monotonic_ns","command_received_ns"),
            ("wait_for_scheduled_tick","command_received_ns","gate_ready_ns"),("physical_x_apply","gate_ready_ns","x_ack_ns"),
            ("capture_to_x_ack","monotonic_ns","x_ack_ns")]
    parts={}
    for label,a,b in fields:
        ms=(values[b]-values[a])/1e6
        parts[label]=dict(zip(("p50_ms","p95_ms","p99_ms","max_ms"),map(float,np.percentile(ms,[50,95,99,100]))))
    arrays=read_records(raw)
    missed=[]
    for i,g in enumerate(gates[:-1]):
        tick=int(g["apply_target"])+1;src=arrays[int(g["stage"])];requested=game_word(int(g["x11_mask"]))
        if int(g["apply_observed"])==int(g["apply_target"]) and np.all(src[tick:tick+4]==requested):continue
        missed.append(dict(index=i,native_tick=tick,scheduled_gate_lateness=int(g["apply_observed"]-g["apply_target"]),
            x11_mask=int(g["x11_mask"]),actual_words=src[tick:tick+4].tolist(),expected_word=requested,
            intervals_ms={n:round(float((values[b][i]-values[a][i])/1e6),4) for n,a,b in fields}))
    movie=json.loads((raw/"movie-pipeline.json").read_bytes())
    # Original writer gets one color frame for each logged original image. An explicitly
    # capped engineering run includes one final captured cap frame without an action log row.
    assert movie["encoded_frames"]>=len(logs)
    return dict(decisions=len(play),source_clock="Single Linux CLOCK_MONOTONIC on the original-game host",
                pipeline=parts,missed=missed,movie=movie,logged_original_frames=len(logs))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--raw",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.write_text(json.dumps(profile(a.raw),indent=2)+"\n",encoding="utf-8")
