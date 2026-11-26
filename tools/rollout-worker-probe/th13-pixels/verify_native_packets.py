"""Reconstruct a sealed shared update from stock returns and per-credit worker score packets."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import zlib

import numpy as np


def verify(directory, parent, candidate):
    rows=json.loads((directory/"credits.json").read_bytes())
    update=json.loads((directory/"training-update.json").read_bytes())
    weights=np.load(parent,allow_pickle=False)["weights"]
    raw_hash=hashlib.sha256(weights.astype("<f8").tobytes()).hexdigest()
    if raw_hash!=update["parent_sha256"]:
        raise ValueError("Incorrect parent checkpoint")
    sums={}
    packed_bytes=0
    raw_bytes=0
    for row in rows:
        packed=(directory/row["packet"]).read_bytes()
        if hashlib.sha256(packed).hexdigest()!=row["packet_sha256"]:
            raise ValueError("Packet SHA failed")
        raw=zlib.decompress(packed)
        n=struct.unpack("<I",raw[:4])[0]
        header=json.loads(raw[4:4+n])
        score=np.frombuffer(raw[4+n:],dtype="<f8").reshape(weights.shape)
        if header["parent"]!=raw_hash or header["title"]!=row["title"] or abs(header["reward"]-row["scaled_return"])>1e-12:
            raise ValueError("Stale policy or incorrect reward in gradient packet")
        if header["score_decisions"]!=row["score_decisions"]:
            raise ValueError("Action boundary differs between packet and native credit")
        if not np.isfinite(score).all():raise ValueError("Nonfinite score gradient")
        sums.setdefault(row["title"],[]).append((header["reward"],score))
        packed_bytes+=len(packed)
        raw_bytes+=header["raw_pixel_bytes"]+header["behavior_bytes"]
    if set(sums)!={12,13} or len(sums[12])!=len(sums[13]):
        raise ValueError("Incomplete two-title barrier")
    per_title=[]
    for entries in sums.values():
        returns=np.array([r for r,_ in entries])
        baseline=(returns.sum()-returns)/(len(returns)-1)
        per_title.append(np.mean((returns-baseline)[:,None,None]*np.stack([s for _,s in entries]),axis=0))
    gradient=np.mean(per_title,axis=0)
    delta=gradient*min(1.,2/np.linalg.norm(gradient))
    reproduced=weights+update["step_size"]*delta
    target=np.load(candidate,allow_pickle=False)["weights"]
    if hashlib.sha256(target.astype("<f8").tobytes()).hexdigest()!=update["weights_sha256"]:
        raise ValueError("Incorrect candidate checkpoint")
    error=float(np.max(np.abs(target-reproduced)))
    if error>1e-12:raise ValueError("Worker packets failed to reproduce the shared update")
    return dict(credits=len(rows),score_packet_bytes=packed_bytes,raw_pixel_and_behavior_bytes=raw_bytes,
                payload_ratio=raw_bytes/packed_bytes,max_abs_parameter_error=error,
                qualification="Worker packet bytes exclude transport framing. The backtracking step size is recorded from the KL test on the archived action features.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ("directory","parent","candidate","out"):parser.add_argument(arg,type=Path)
    args=parser.parse_args()
    args.out.write_text(json.dumps(verify(args.directory,args.parent,args.candidate),indent=2)+"\n",encoding="utf-8")
