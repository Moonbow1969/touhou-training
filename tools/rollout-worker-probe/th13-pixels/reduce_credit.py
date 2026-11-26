"""Reduce a native actor's pixel trace to one score-function packet without ML dependencies."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import zlib


FRAME=112*96
WIDTH=85
WEIGHTS=struct.Struct("<255d")


def feature(frame,previous):
    current=[]
    motion=[]
    for y in range(0,112,16):
        for x in range(0,96,16):
            signal=change=0
            for row in range(y,y+16):
                i=row*96+x
                now=frame[i:i+16]
                before=previous[i:i+16]
                signal+=sum(now)
                change+=sum(abs(a-b) for a,b in zip(now,before))
            current.append(signal/(255.*256))
            motion.append(change/(255.*256))
    return [1.,*current,*motion]


def acts(path,game):
    if game==13:
        with path.open() as stream:
            for row in csv.DictReader(stream,delimiter="\t"):
                yield int(row["action"]),[float(row[k]) for k in ("p_left","p_neutral","p_right")],float(row["logp"])
    else:
        for line in path.read_text().splitlines():
            fields=line.split()
            if fields and fields[0] in {"0","1","2"}:
                yield int(fields[0]),list(map(float,fields[5:8])),float(fields[8])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--title",type=int,choices=[12,13],required=True)
    parser.add_argument("--pixels",type=Path,required=True)
    parser.add_argument("--actions",type=Path,required=True)
    parser.add_argument("--weights",type=Path,required=True)
    parser.add_argument("--prefix-steps",type=int,required=True)
    parser.add_argument("--reward",type=float,required=True)
    parser.add_argument("--expected-parent",required=True)
    args=parser.parse_args()
    raw=args.weights.read_bytes()
    fingerprint=hashlib.sha256(raw).hexdigest()
    if fingerprint!=args.expected_parent:
        raise ValueError("Stale actor weights.")
    weights=WEIGHTS.unpack(raw)
    grad=[0.]*255
    stream=list(acts(args.actions,args.title))
    if not 1<=args.prefix_steps<=len(stream) or args.pixels.stat().st_size!=(len(stream)+1)*FRAME:
        raise ValueError("Invalid native action or pixel length.")
    previous=None
    with args.pixels.open("rb") as frames:
        for i,(action,actual,logp) in enumerate(stream):
            frame=frames.read(FRAME)
            vec=feature(frame,frame if previous is None else previous)
            previous=frame
            logits=[sum(w*x for w,x in zip(weights[k*WIDTH:(k+1)*WIDTH],vec)) for k in range(3)]
            exp=[math.exp(x-max(logits)) for x in logits]
            probs=[v/sum(exp) for v in exp]
            if max(abs(x-y) for x,y in zip(probs,actual))>4e-9 or abs(math.log(probs[action])-logp)>3e-8:
                raise ValueError("Logged behavior differs from the loaded policy.")
            if i<args.prefix_steps:
                for k,p in enumerate(probs):
                    factor=float(k==action)-p
                    for j,value in enumerate(vec,start=k*WIDTH):
                        grad[j]+=factor*value
    if not math.isfinite(args.reward) or abs(args.reward)>1.000001:
        raise ValueError("Invalid normalized credit return.")
    meta=dict(title=args.title,parent=fingerprint,decisions=len(stream),score_decisions=args.prefix_steps,
              reward=args.reward,raw_pixel_bytes=args.pixels.stat().st_size,behavior_bytes=args.actions.stat().st_size)
    head=json.dumps(meta,sort_keys=True,separators=(",",":")).encode()
    sys.stdout.buffer.write(zlib.compress(struct.pack("<I",len(head))+head+WEIGHTS.pack(*grad)))


if __name__=="__main__":
    main()
