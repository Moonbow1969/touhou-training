"""Join privileged TH08 coordinates to the observed pre-action camera for an offline control."""
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


def _source(manifest,row):
    directory=Path(manifest).resolve().parent
    receipt_file=directory/row["receipt"]
    receipt_bytes=receipt_file.read_bytes()
    if hashlib.sha256(receipt_bytes).hexdigest()!=row["receipt_sha256"]:
        raise ValueError("TH08 registered receipt checksum differs")
    receipt=json.loads(receipt_bytes)
    if (receipt["role"]!="train" or not receipt["qualified"] or receipt["seed"]!=row["seed"] or
            receipt["policy_sha256"]!=row["policy_sha256"]):
        raise ValueError("Expected a qualified original training credit for this behavior policy")
    if row.get("archive"):
        with ZipFile(directory/row["archive"]) as archive:
            raw={f:archive.read(f) for f in ("gates.tsv","clock.tsv")}
    else:
        folder=receipt_file.parent.parent/"raw"/str(row["seed"])
        raw={f:(folder/f).read_bytes() for f in ("gates.tsv","clock.tsv")}
    for f,k in (("gates.tsv","input_gates_sha256"),("clock.tsv","stage_clock_trace_sha256")):
        if hashlib.sha256(raw[f]).hexdigest()!=receipt[k]:raise ValueError(f"TH08 source {f} changed")
    return receipt,raw


def read_camera_context(manifest:Path,row:dict,ticks:np.ndarray):
    """Return causal current XY/state plus the separate, label-only original state timeline."""
    receipt,raw=_source(manifest,row)
    gates=pd.read_csv(io.BytesIO(raw["gates.tsv"]),sep=r"\s+")
    clock=pd.read_csv(io.BytesIO(raw["clock.tsv"]),sep=r"\s+")
    n=receipt["decisions"]
    ticks=np.asarray(ticks)
    recorded_requests=receipt.get("physical_sampled_requests",n)
    if (ticks.shape!=(n,) or len(gates)!=recorded_requests or not np.array_equal(gates["index"],np.arange(len(gates)))
            or not np.array_equal(gates["apply_target"].to_numpy()[:n]+1,ticks)):
        raise ValueError("TH08 optimizer decision rows do not match the chronological physical gates")
    if len(gates)!=n:
        trim=receipt.get("excluded_terminal_proposal")
        if len(gates)!=n+1 or not trim or trim.get("native_exposure")!=0:
            raise ValueError("Unqualified physical surplus outside the sole signed post-terminal proposal")
        gates=gates.iloc[:n]
    live=(clock["stable"]==1)&(clock["mode"]==0)&(clock["flags"]==4)&(clock["object"]!=0)&(clock["origin"]!=0)&(clock["buffer_read"]==1)
    live&=(clock["cursor"]==clock["origin"]+36+2*clock["tick"])&(clock["saved_current"]==clock["word"])
    live&=(clock["stage_index"]==0)&(clock["tick"]>0)&(clock["before_ns"]<=clock["after_ns"])
    live&=clock["x"].between(0,384)&clock["y"].between(0,464)&((clock["x"]!=0)|(clock["y"]!=0))
    observed=clock.loc[live,["tick","before_ns","after_ns","player","focus","x","y"]]
    if observed.empty or not (gates["stage"]==1).all() or not observed["after_ns"].is_monotonic_increasing:
        raise ValueError("Missing or nonchronological TH08 Stage 1 observer")
    # The stream polls the native tick, calls XGetImage, then records camera_ns.
    # The first stable read for that same tick was already available before capture.
    # Never borrow a coordinate from a later native tick, even if close in wall time.
    first=observed.drop_duplicates("tick",keep="first").set_index("tick")
    sampled=first.reindex(gates["camera_observed"].to_numpy())
    known=sampled["after_ns"].notna().to_numpy()&(sampled["after_ns"].to_numpy()<=gates["camera_ns"].to_numpy())
    if not known.all():
        raise ValueError(f"TH08 seed {row['seed']} has {known.sum()}/{n} exactly timed source camera coordinates")
    camera=dict(xy=sampled[["x","y"]].to_numpy(dtype=np.float32),player=sampled["player"].to_numpy(dtype=np.uint8),
                focus=sampled["focus"].to_numpy(dtype=np.uint8),tick=gates["camera_observed"].to_numpy(),
                camera_ns=gates["camera_ns"].to_numpy(),observer_after_ns=sampled["after_ns"].to_numpy(dtype=np.int64))
    return camera,observed


def read_coordinates(manifest:Path,row:dict,ticks:np.ndarray)->np.ndarray:
    """Original x,y in the native 384×448 playfield coordinate system, one row per action."""
    camera,_=read_camera_context(manifest,row,ticks)
    return camera["xy"]


def read_native_camera(manifest:Path,row:dict,ticks:np.ndarray,stock_event_ticks:np.ndarray)->dict:
    """Join causal positions and vulnerable camera pairs; resolve retrospective, stock-confirmed hit onsets."""
    camera,observed=read_camera_context(manifest,row,ticks)
    receipt=json.loads((Path(manifest).parent/row["receipt"]).read_bytes())
    events=receipt["life_events"]
    declared=np.asarray(stock_event_ticks)
    if declared.shape!=(3,) or len(events)!=3 or any(not e["native_tick"]<=t<=e["next_tick"] for e,t in zip(events,declared)):
        raise ValueError("Expected the three original stock-event ticks within their signed HUD intervals")
    lo=camera["tick"][0]-5;hi=receipt["terminal_tick"]
    within=observed[(observed["tick"]>=lo)&(observed["tick"]<=hi)]
    tick=within["tick"].to_numpy(dtype=np.int64)
    state=within["player"].to_numpy(dtype=np.int64)
    changes=np.flatnonzero(state[1:]!=state[:-1])+1
    deaths=[]
    recovered=[]
    recoveries=[]
    for j,i in enumerate(changes):
        if state[i]!=2:continue
        if j+1>=len(changes):raise ValueError("Incomplete original hit-state episode through terminal")
        after=changes[j+1]
        if state[after]==3:
            recovered.append(int(tick[i]))
            recoveries.append(dict(tick=int(tick[i]),prior_state=int(state[i-1]),recovered_state=3,recovered_tick=int(tick[after])))
            continue
        if state[after]!=1 or state[i-1] not in (0,3) or tick[i]-tick[i-1]!=1 or not 29<=tick[after]-tick[i]<=32:
            raise ValueError(f"Unresolved original player-state transition at tick {tick[i]} for {row['seed']}")
        if state[i-1]==3:
            # A hit can be taken on the exact frame the long invulnerability period ends.
            if j==0 or tick[i]-tick[changes[j-1]]<200:
                raise ValueError("Unexpected short invulnerable-state to hit transition")
        deaths.append(dict(tick=int(tick[i]),previous=int(state[i-1]),last_pre_hit_tick=int(tick[i-1]),
                           last_pre_hit_after_ns=int(within.iloc[i-1]["after_ns"]),first_hit_before_ns=int(within.iloc[i]["before_ns"]),
                           first_hit_after_ns=int(within.iloc[i]["after_ns"]),stock_transition_tick=int(tick[after])))
    if len(deaths)!=3 or any(not e["native_tick"]-9<=hit["stock_transition_tick"]<=e["next_tick"] for e,hit in zip(events,deaths)):
        raise ValueError(f"The observed state episodes do not uniquely reconcile with the original stock ledger for {row['seed']}")
    current=camera["player"]==0
    vulnerable=current&np.r_[current[0],current[:-1]]
    # Also reject any nonvulnerable native observation BETWEEN these two causal images.
    if not np.all(np.diff(tick)>=0):
        raise ValueError("Original stage clock regressed within admitted stock")
    previous_camera=np.r_[camera["tick"][0],camera["tick"][:-1]]
    clock_seen=within["after_ns"].to_numpy(dtype=np.int64)
    unsafe_times=clock_seen[state!=0]
    previous_timestamp=np.r_[camera["observer_after_ns"][0],camera["observer_after_ns"][:-1]]
    vulnerable&=np.searchsorted(unsafe_times,camera["observer_after_ns"],side="right")==np.searchsorted(unsafe_times,previous_timestamp,side="left")
    unique_ticks=np.unique(tick)
    count=np.searchsorted(unique_ticks,camera["tick"],side="right")-np.searchsorted(unique_ticks,previous_camera,side="left")
    vulnerable&=count==camera["tick"]-previous_camera+1
    return dict(xy=camera["xy"],vulnerable=vulnerable,hit_ticks=np.array([d["tick"] for d in deaths],dtype=np.int64),
                camera_ticks=camera["tick"],camera_player_state=camera["player"],
                camera_ns=camera["camera_ns"],coordinate_after_ns=camera["observer_after_ns"],
                verified_hit_transitions=deaths,recovered_hit_ticks=np.asarray(recovered,dtype=np.int64),recovered_hit_transitions=recoveries)
