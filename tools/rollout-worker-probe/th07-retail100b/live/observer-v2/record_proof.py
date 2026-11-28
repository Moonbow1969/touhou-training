"""Check the independent retail result glyphs and the sequenced asynchronous evidence."""
import csv
import hashlib
import json
from pathlib import Path
import struct
import zlib

import numpy as np
from PIL import Image


def read_score(picture, template):
    pixels = np.array(Image.open(picture).convert("RGB")).min(-1) > 185
    masks = {k: [np.unpackbits(np.frombuffer(bytes.fromhex(x), dtype=np.uint8))[:252].reshape(18,14).astype(bool)
                 for x in glyphs] for k,glyphs in template["packed_masks"].items()}
    digits = []
    confidence = []
    for i in range(10):
        patch = pixels[126:144,385-14*i:399-14*i]
        if patch.sum() < 7:
            break
        scores = sorted(((max((patch & glyph).sum()/(patch | glyph).sum() for glyph in glyphs), name)
                        for name,glyphs in masks.items()), reverse=True)
        if scores[0][0]<.75 or scores[0][0]-scores[1][0]<.055:
            raise ValueError(f"Uncertain score digit {i}: {scores[:2]}")
        digits.append(scores[0][1])
        confidence.append(float(scores[0][0]))
    if not digits:
        raise ValueError("Missing result screen score")
    return int("".join(reversed(digits))), min(confidence)


def official_score(folder, replay_score, last_live):
    font=json.loads((Path(__file__).parent/"score-font.json").read_bytes())
    first=read_score(folder/"settled-scorecard.png",font)
    second=read_score(folder/"settled-scorecard-repeat.png",font)
    return dict(score_confirmed=first[0]==second[0]==replay_score, screen1=first[0], screen2=second[0],
                saved_rpy=int(replay_score), last_live=int(last_live), settlement_delta=int(replay_score-last_live),
                minimum_glyph_iou=min(first[1],second[1]))


def verify_archive(folder):
    n=int((folder/"archive.sealed").read_text())
    files=[folder/"native-input.csv",folder/"applied.csv"]
    expected=[f.read_bytes().splitlines(keepends=True)[1:] for f in files]
    positions=[0,0]
    payload=(folder/"events.bin").read_bytes()
    p=0
    for seq in range(n):
        if p+16>len(payload):raise ValueError("Truncated journal header")
        i,lane,length=struct.unpack_from("<III",payload,p)
        if i!=seq or lane>1 or not 0<length<256:raise ValueError("Invalid sequence, kind or length")
        end=p+12+length
        if end+4>len(payload) or zlib.crc32(payload[p:end])!=struct.unpack_from("<I",payload,end)[0]:
            raise ValueError("Invalid journal CRC")
        if positions[lane]>=len(expected[lane]) or payload[p+12:end]!=expected[lane][positions[lane]]:
            raise ValueError("Evidence files disagree")
        positions[lane]+=1
        p=end+4
    if p!=len(payload) or positions!=list(map(len,expected)):
        raise ValueError("Invalid sealed tail")
    originals=list(csv.DictReader(files[0].read_text().splitlines()))
    seen={}
    duplicates=0
    for r in originals:
        tick,ix,clock=map(int,(r["tick"],r["event_index"],r["stageclock"]))
        if tick!=ix or tick!=clock or r["word"]!=r["record_slot"] or (tick and r["previous_word"]!=r["previous_slot"]):
            raise ValueError("Inconsistent native clock, cursor or record words")
        key=(int(r["stage"]),tick)
        signature=tuple(r[k] for k in ("word","previous_word","rank","shot","continues","lives","bombs","misses","bombs_used"))
        if key in seen:
            if signature!=seen[key]:raise ValueError("Conflicting repeated native sample")
            duplicates+=1
        seen[key]=signature
    return dict(records=n,native_rows=positions[0],applied_rows=positions[1],stable_duplicate_rows=duplicates,
                journal_sha256=hashlib.sha256(payload).hexdigest())
