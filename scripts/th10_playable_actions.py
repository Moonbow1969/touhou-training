"""Read-only evidence for a playable TH10 training prefix at actual native Game Over."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def verify_source(source, *, max_tail=1):
    """Accept an original report.json or its directory; return the physically qualified exclusive end."""
    file=Path(source)
    if file.is_dir():
        file=file/"report.json"
    folder=file.parent
    receipt=json.loads(file.read_bytes())
    behavior=folder/"behavior.npz"
    native=folder/"native.csv.gz"
    if not receipt["terminal"] or receipt["continues"] or digest(behavior)!=receipt["behavior_sha256"]:
        raise ValueError(f"Missing complete original TH10 source: {folder}")
    with gzip.open(native,"rt") as f:
        samples=list(csv.DictReader(f))
    stock=np.array([int(r["lives"]) for r in samples])
    utc=np.array([int(r["utc_ms"]) for r in samples],dtype=np.int64)
    if (len(utc)<2 or np.any(np.diff(utc)<0) or list(stock[-2:])!=[0,-1]
            or np.any(stock[:-1]<0) or len(utc)!=receipt["native_samples"]):
        raise ValueError(f"Unqualified native terminal: {folder}")
    native_end=int(utc[-1])
    with np.load(behavior) as b:
        actions=b["action_utc_ms"]
        n=len(actions)
        if not (np.all(np.diff(actions)>0) and n==receipt["steps"] and len(b["reward"])==n
                and b["terminated"].sum()==1 and b["terminated"][-1]):
            raise ValueError(f"Unqualified action time base: {folder}")
        end=int(actions.searchsorted(native_end,side="right"))
        if end==0 or (max_tail is not None and n-end>max_tail) or b["reward_life"][end-1]!=-1:
            raise ValueError(f"Unqualified unsigned or unaligned fatal step: {folder}")
        if n>end:
            if (b["native_samples"][end-1]<=0 or b["native_input_matches"][end-1]<=0
                    or np.any(b["native_samples"][end:]) or np.any(b["native_input_matches"][end:])):
                raise ValueError(f"Tail actions include a live input or lack signed preceding input: {folder}")
            for key in ("reward","reward_life","reward_survival","reward_stage","reward_bomb"):
                if np.any(b[key][end:]!=0):
                    raise ValueError(f"Nonzero {key} on the native-terminal tail in {folder}")
        return dict(seed=receipt["seed"],rank=receipt["difficulty"],role=receipt.get("role"),policy_sha256=receipt["policy_sha256"],
            original_steps=n,playable_steps=end,terminal_action_index=end-1,dropped_steps=n-end,
            logged_terminal_input_utc_ms=int(actions[-1]),native_death_utc_ms=native_end,
            last_live_utc_ms=int(utc[-2]),first_removed_action_minus_native_terminal_ms=int(actions[end]-native_end) if end<n else None,
            native_terminal_input_word=int(samples[-1]["native_input"]),fatal_action_readings=int(b["native_samples"][end-1]),
            fatal_action_native_matches=int(b["native_input_matches"][end-1]),original_done_index=n-1,return_sum=float(b["reward"].sum(dtype=np.float64)),
            trace_sha256=digest(behavior),native_csv_sha256=digest(native),report_sha256=digest(file))


def crop_packed(arrays, evidence):
    """Return copies of aligned packed inputs and move done to the retained fatal action."""
    total,end=evidence["original_steps"],evidence["playable_steps"]
    result={}
    for name,array in arrays.items():
        a=np.asarray(array)
        if a.ndim==0 or len(a)!=total:
            raise ValueError(f"Unaligned TH10 packed array {name}, expected {total} rows")
        result[name]=a[:end].copy()
    if not np.array_equal(np.flatnonzero(arrays["terminated"]),[total-1]):
        raise ValueError("Packed TH10 source has more than one terminal or it is misplaced")
    if not np.array_equal(np.flatnonzero(result["terminated"]),[] if end<total else [end-1]):
        raise ValueError("Inconsistent retained TH10 terminal marker")
    if np.any(np.asarray(arrays["reward"])[end:]!=0):
        raise ValueError("Cannot discard a nonzero packed final reward")
    result["terminated"][-1]=True
    return result
