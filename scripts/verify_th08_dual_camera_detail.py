"""Measure real spatial variation inside each 2×2 high-resolution cell aligned to a policy pixel."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def measure(study,row):
    larger=study/row["highres"];smaller=study/row["file"]
    if digest(larger)!=row["highres_sha256"] or digest(smaller)!=row["sha256"]:
        raise ValueError("Mismatched aligned camera source")
    with np.load(larger,allow_pickle=False) as z:big=z["gray"]
    with np.load(smaller,allow_pickle=False) as z:little=z["frames"]
    idx=np.asarray(row["original_observation_indices"],dtype=np.int64)
    if not np.all(np.diff(idx)>0) or len(idx)!=len(little):raise ValueError("Physical observations not strictly aligned")
    histogram=np.zeros(256,dtype=np.int64);total=0;maximum=0;two_level=0
    for i in range(0,len(idx),32):
        section=slice(i,i+32)
        cells=big[idx[section]].reshape(-1,112,2,96,2)
        spread=cells.max((2,4))-cells.min((2,4))
        histogram+=np.bincount(spread.ravel(),minlength=256)
        reduced=(cells.sum((2,4)).astype(np.int32)+2)//4
        delta=np.abs(reduced-little[section])
        total+=int(delta.sum());maximum=max(maximum,int(delta.max()));two_level+=int(np.count_nonzero(delta==2))
    return dict(seed=row["seed"],observations=len(idx),policy_pixels=int(histogram.sum()),nonconstant_highres_cells=int(histogram[1:].sum()),
        cells_with_contrast_8_or_more=int(histogram[8:].sum()),cells_with_contrast_32_or_more=int(histogram[32:].sum()),
        maximum_luminance_disagreement=maximum,two_level_disagreement_pixels=two_level,mean_absolute_luminance_disagreement=total/int(histogram.sum()),
        highres_sha256=row["highres_sha256"],lowres_sha256=row["sha256"])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("study",type=Path);parser.add_argument("out",type=Path)
    args=parser.parse_args()
    manifest=args.study/"qualified-train.json"
    inputs=[json.loads(p.read_bytes()) for p in sorted((args.study/"native08/selected").glob("*.json"))]
    info=[measure(args.study,r) for r in inputs]
    pixels=sum(r["policy_pixels"] for r in info)
    result=dict(source_manifest_sha256=digest(manifest) if manifest.exists() else None,
        physical=len(info),registered_original_cells=pixels,
        nonconstant_highres_cell_fraction=sum(r["nonconstant_highres_cells"] for r in info)/pixels,
        highres_contrast_8_fraction=sum(r["cells_with_contrast_8_or_more"] for r in info)/pixels,
        highres_contrast_32_fraction=sum(r["cells_with_contrast_32_or_more"] for r in info)/pixels,
        worst_low_resolution_alignment_error=max(r["maximum_luminance_disagreement"] for r in info),two_level_disagreement_pixels=sum(r['two_level_disagreement_pixels'] for r in info),
        meaning="Fractions of physical policy pixels whose four native high-resolution grayscale subpixels differ by at least the stated contrast on the 0–255 scale.",
        per_credit=info)
    args.out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
