"""Inspect the original TH08 results table after policy input has ended."""
from pathlib import Path
import cv2
import numpy as np


def centered(im):
    im=im.astype(np.float32)
    return im-cv2.GaussianBlur(im,(0,0),1.1)


def correlation(sample,ref):
    mask=np.abs(ref)>14
    x,r=sample[mask],ref[mask]
    return float(r@x/max(np.linalg.norm(r)*np.linalg.norm(x),1e-9))


def inspect_results(folder):
    reference=Path(__file__).resolve().parents[1]/'tools/rollout-worker-probe/th08-trial/score-row-stage-one.npz'
    with np.load(reference,allow_pickle=False) as src:
        header,date,one=(src[k] for k in ('header','date','one'))
    candidates=[]
    for f in sorted(folder.glob('post-credit-*.png')):
        im=cv2.imread(str(f),cv2.IMREAD_GRAYSCALE)
        if im is None:continue
        label=correlation(centered(im[124:145,248:405]),header)
        if label<.80:continue
        scores=[]
        for i in range(10):
            y=156+18*i
            stamp=correlation(centered(im[y-8:y+9,437:510]),date)
            digit=correlation(centered(im[y-8:y+9,366:378]),one)
            scores.append((stamp,digit,i+1))
        scores.sort(reverse=True)
        stamp,digit,row=scores[0]
        if stamp<.75 or stamp-scores[1][0]<.30:continue
        candidates.append(dict(file=f.name,header=label,date=stamp,date_margin=stamp-scores[1][0],row=row,one=digit))
    if not candidates:
        return dict(stage=None,reason='No unique dated entry in the original results table',observations=[])
    consistent={c['row'] for c in candidates}
    stage=1 if len(candidates)>=2 and len(consistent)==1 and min(c['one'] for c in candidates)>.88 else None
    return dict(stage=stage,observations=candidates)
