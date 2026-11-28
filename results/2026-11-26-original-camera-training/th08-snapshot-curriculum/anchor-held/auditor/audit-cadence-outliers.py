"""Locate camera gate misses in the native observer, freeze and actor clocks."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import numpy as np


def run(study,dest):
    rows=[]
    for path in sorted((study/"receipts").glob("*.json")):
        receipt=json.loads(path.read_bytes())
        bad=receipt["control_evidence"]["short_partial_first_slots"]
        if not bad:
            continue
        archive=study/"evidence"/f"{receipt['seed']}-compact.tar.gz"
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==receipt["compact_archive_sha256"]
        with tarfile.open(archive)as source:
            def table(filename):
                return np.genfromtxt(io.BytesIO(source.extractfile(filename).read()),names=True,dtype=None,encoding="ascii",ndmin=1)
            c,g,f=table("clock.tsv"),table("gates.tsv"),table("frozen.tsv")
            for miss in bad:
                i=receipt["offset"]+miss["index"]
                gate=g[i]
                future=g[i+1] if i+1<len(g) else None
                row=dict(seed=receipt["seed"],arm=receipt["arm"],decision=miss["index"],
                         requested_word=miss["expected"],consumed_words=miss["actual"],
                         start_tick=int(gate["camera_observed"]),end_tick=int(gate["camera_observed"])+len(miss["actual"]),
                         source_sha256=receipt["compact_archive_sha256"])
                if future is not None:
                    freeze=f[i+1]
                    t=int(future["camera_target"]);stage=int(future["stage"])
                    samples=c[(c["stage_index"]==stage-1)&(c["tick"]==t)&(c["flags"]==4)&(c["stable"]==1)]
                    ms=lambda a,b:(int(a)-int(b))/1e6
                    row.update(next_requested_camera_tick=t,next_observed_camera_tick=int(future["camera_observed"]),
                               observed_stable_target_samples=len(samples),max_target_read_ms=(
                                   max(ms(v["after_ns"],v["before_ns"]) for v in samples) if len(samples) else None),
                               ms_from_first_target_read_return_to_next_stop=(ms(freeze["stop_start_ns"],samples[0]["after_ns"]) if len(samples) else None),
                               next_camera_to_command_ms=ms(freeze["cmd_ns"],freeze["camera_ns"]),
                               next_command_to_x_ack_ms=ms(freeze["ack_ns"],freeze["cmd_ns"]),
                               next_stop_to_resume_ms=ms(freeze["resume_ns"],freeze["stop_start_ns"]))
                rows.append(row)
    dest.write_text(json.dumps(dict(nonexact_held_windows=rows),indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("study",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.study,a.out)
