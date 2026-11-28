"""Seal an original TH07 retail 1.00b human replay stage and build bounded causal windows."""
from pathlib import Path
import sys, struct, hashlib, json, gzip
import numpy as np
from PIL import Image
from inspect_rpy import decompress

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def source_info(path, stage):
    src=bytearray(path.read_bytes())
    assert src[:6]==bytes.fromhex('543752500011')
    src[16:]=bytes((v-src[13]-7*i)%256 for i,v in enumerate(src[16:]))
    clen,dlen=struct.unpack_from('<II',src,20)
    assert len(src)==84+clen
    data=decompress(bytes(src[84:]),dlen)
    off=struct.unpack_from('<14I',src,28);start=off[stage-1]
    end=min(v for v in [*off,84+len(data)] if v>start)
    score=int.from_bytes(data[24:28],'little')*10
    assert data[1:4]==b'b\x00\x00'and data[140:146]==b'0100b\x00'and int.from_bytes(data[22:24],'little')==256
    assert int.from_bytes(data[132:136],'little')==650752 and int.from_bytes(data[136:140],'little')==0xaec5445c
    real=np.frombuffer(bytes(data[start-84+44:end-84]),'<u2').reshape(-1,2)
    snapshot=np.frombuffer(bytes(data[start-84:start-84+44]),'u1')
    return real[:,0],snapshot,score


