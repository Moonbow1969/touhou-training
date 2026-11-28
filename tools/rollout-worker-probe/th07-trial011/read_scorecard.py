"""Read original TH07 result digits using independently transcribed previous full-credit screenshots."""
from pathlib import Path
import json
import numpy as np
from PIL import Image


def tile(image, x, y):
    a=np.asarray(image.crop((x-7,y-8,x+7,y+9)).convert("L"))>166
    return a


def glyph_bank(archive, readings, omit=None):
    bank={k:[]for k in range(10)}
    for run,truth in readings.items():
        if run==omit:continue
        img=Image.open(archive/run/"settled-scorecard.png")
        for y,text in [(135,str(truth["score"])),(201,str(truth["continue"])),(223,str(truth["miss"])),(245,str(truth["bombs"]))]:
            for i,ch in enumerate(text[::-1]):
                bank[int(ch)].append(tile(img,392-14*i,y))
    return bank


def read_int(image,bank,y,offsets=tuple(range(9))):
    digits=[];lowest=1.;margins=[]
    for j in offsets:
        a=tile(image,392-14*j,y)
        if a.sum()<5:break
        similarities=[]
        for digit, examples in bank.items():
            if not examples:continue
            best=max(float((a&b).sum()/(a|b).sum())for b in examples)
            similarities.append((best,digit))
        ranks=sorted(similarities,reverse=True)
        digits.append(str(ranks[0][1]));lowest=min(lowest,ranks[0][0]);margins.append(ranks[0][0]-ranks[1][0])
    if not digits:return None,0.,0.
    return int("".join(reversed(digits))),lowest,min(margins)


def scorecard(path, archive, reference):
    bank=glyph_bank(archive,reference)
    img=Image.open(path)
    read={k:read_int(img,bank,y)for k,y in (("score",135),("continue",201),("miss",223),("bombs",245))}
    val,sim,margin=read_int(img,bank,179,(1,2,4,5,6))
    read["progress_pct"]=(val/100 if val is not None else None,sim,margin)
    return read


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("archive",type=Path);p.add_argument("reference",type=Path);a=p.parse_args()
    labels=json.loads(a.reference.read_bytes())
    for credit,v in labels.items():
        bank=glyph_bank(a.archive,labels,omit=credit)
        for field,y in (("score",135),("continue",201),("miss",223),("bombs",245)):
            got=read_int(Image.open(a.archive/credit/"settled-scorecard.png"),bank,y)
            print(credit,field,got,v[field],got[0]==v[field])
        got=read_int(Image.open(a.archive/credit/"settled-scorecard.png"),bank,179,(1,2,4,5,6))
        print(credit,"progress_pct",got,v["progress_pct"],got[0]/100==v["progress_pct"])
