"""Audit exact native hit onsets, later HUD loss frames, causal coordinates, and recovered hits."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from scripts.th08_hazard_coordinates import read_native_camera


def sha(path):
    with path.open("rb") as handle:return hashlib.file_digest(handle,"sha256").hexdigest()


def run(data):
    rows=[]
    for group,relative in (("shared_sft","th08-progress-reward/qualified-train.json"),
                           ("lambda1_ppo","th08-progress-lambda1-round2/qualification-diagnostic.json")):
        manifest=data/relative;directory=manifest.parent
        for source in json.loads(manifest.read_bytes())["rows"]:
            if not source.get("qualified",True):continue
            receipt_file=directory/source["receipt"];receipt=json.loads(receipt_file.read_bytes())
            shard=directory/source["file"]
            if sha(shard)!=source["sha256"] or sha(receipt_file)!=source["receipt_sha256"]:raise ValueError("Physical source hash changed")
            if source.get("archive"):
                with ZipFile(directory/source["archive"]) as z:timing=z.read("timing.jsonl")
                source_name=f'{directory.name}/{source["archive"]}'
            else:
                raw=receipt_file.parent.parent/"raw"/str(source["seed"])
                timing=(raw/"timing.jsonl").read_bytes();source_name=raw.relative_to(data).as_posix()
            if hashlib.sha256(timing).hexdigest()!=receipt["timing_sha256"]:raise ValueError("Signed GUI timings changed")
            physical=[json.loads(line) for line in timing.splitlines()]
            active=[g for g in physical if g["phase"]=="policy"]
            with np.load(shard,allow_pickle=False) as z:
                ticks=z["native_tick"];ends=z["native_next_tick"];idx=np.flatnonzero(z["life_loss"])
            native=read_native_camera(manifest,source,ticks,ends[idx])
            if not np.array_equal([g["monotonic_ns"] for g in active[:len(ticks)]],native["camera_ns"]):raise ValueError("Physical game camera identities differ")
            events=[]
            for j,(i,hud,hit) in enumerate(zip(idx,receipt["life_events"],native["verified_hit_transitions"])):
                a=active[i]
                b=active[i+1] if j<2 else physical[-1]
                if j==2 and b["phase"]!="terminal":raise ValueError("Missing genuine original terminal image")
                events.append(dict(ordinal=j+1,native_hit=hit,original_hud_event=hud,
                    actual_gui_before=dict(phase=a["phase"],frame=a["frame"],native_camera_tick=a["camera_native_tick"],monotonic_ns=a["monotonic_ns"],spares=a["spares"],bombs=a["bombs"]),
                    actual_gui_after=dict(phase=b["phase"],frame=b["frame"],native_camera_tick=b.get("camera_native_tick"),monotonic_ns=b["monotonic_ns"],spares=b["spares"],bombs=b["bombs"]),
                    signed_interval_start_minus_hit_ticks=int(ticks[i]-hit["tick"]),
                    signed_interval_end_minus_hit_ticks=int(ends[i]-hit["tick"])))
            latency=(native["camera_ns"]-native["coordinate_after_ns"])/1e6
            rows.append(dict(seed=source["seed"],source_group=group,policy_sha256=source["policy_sha256"],manifest=relative,manifest_sha256=sha(manifest),
                original=source_name,original_archive_sha256=source.get("archive_sha256"),receipt=receipt_file.relative_to(data).as_posix(),
                receipt_sha256=source["receipt_sha256"],optimizer_sha256=source["sha256"],stage_clock_sha256=receipt["stage_clock_trace_sha256"],
                original_gate_sha256=receipt["input_gates_sha256"],gui_timing_sha256=receipt["timing_sha256"],original_executable_sha256=receipt["game_sha256"],
                incoming_frames=len(ticks),causally_joined_camera_rows=len(native["xy"]),vulnerable_two_image_rows=int(native["vulnerable"].sum()),
                trimmed_terminal_physical_requests=len(active)-len(ticks),first_coordinate_read_to_camera_return_ms=dict(median=float(np.median(latency)),p99=float(np.percentile(latency,99)),maximum=float(np.max(latency))),
                deaths=events,recovered_hits=native["recovered_hit_transitions"]))
    recovery=[dict(seed=r["seed"],**e) for r in rows for e in r["recovered_hits"]]
    events=[e for r in rows for e in r["deaths"]]
    if len(set(r["seed"] for r in rows))!=45 or len(events)!=135 or len(recovery)!=5:raise ValueError("Incomplete original source grouping")
    return dict(edition="TH08 official 0.02 trial, Normal Ghost",source_role="train",scope="Independent original data audit; no held outcome or action-policy fit",
        native_label="Distinct 0→2 or long 3→2 onset on consecutive native ticks, followed about 31 ticks later by 2→1 and a unique signed original stock loss or Game Over.",
        camera="The native current stage/tick is polled before XGetImage. camera_ns is recorded immediately on XGetImage return. Privileged XY is the first source-confirmed stable read of the same native tick whose after_ns does not exceed camera_ns. This is a host native tick bound, not a display scanout claim.",
        hud="HUD reserves are sampled from the original right panel (x 496–608); the encoder only receives the original 384×448 playfield x=32..415. The signed loss window refers to the native control interval. actual_gui_before/after are the separately recorded camera samples. A terminal image can carry the last published active frame tick after the original has stopped.",
        rows=rows,total_credits=len(rows),native_hits=len(events),native_0_to_2=sum(e["native_hit"]["previous"]==0 for e in events),native_long_3_to_2=sum(e["native_hit"]["previous"]==3 for e in events),
        nonfatal_recoveries=recovery,total_original_camera_rows=sum(r["incoming_frames"] for r in rows),
        total_two_vulnerable_images=sum(r["vulnerable_two_image_rows"] for r in rows),source_groups={g:{"credits":sum(r["source_group"]==g for r in rows),
                "camera_frames":sum(r["incoming_frames"] for r in rows if r["source_group"]==g),
                "two_vulnerable_images":sum(r["vulnerable_two_image_rows"] for r in rows if r["source_group"]==g)} for g in ("shared_sft","lambda1_ppo")})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("data",type=Path);parser.add_argument("out",type=Path)
    args=parser.parse_args();result=run(args.data)
    text=json.dumps(result,indent=2,ensure_ascii=False)+"\n"
    if args.out.exists() and args.out.read_text(encoding="utf8")!=text:raise ValueError("The existing physical audit differs")
    args.out.parent.mkdir(exist_ok=True,parents=True)
    args.out.write_text(text,encoding="utf8")
    print(result["source_groups"],result["nonfatal_recoveries"])
