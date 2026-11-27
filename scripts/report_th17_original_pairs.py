"""Summarize complete registered TH17 original credit cohorts separately for each native rank."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np

def summarize(rows):
    n=len(rows)
    stage=sum(r["stage2_reached"] for r in rows)
    p=stage/n
    z=1.959963984540054
    delta=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n));denom=1+z*z/n
    valid=[max(0,(p+z*z/(2*n)-delta)/denom),min(1,(p+z*z/(2*n)+delta)/denom)]
    actions=sum(r["actions"] for r in rows)
    moves=np.array([r["action_counts"] for r in rows]).sum(0)
    fields=["seconds","last_stage_native_tick","first_loss_native_tick","remaining_spare_lives","last_live_bombs","remaining_bombs","actual_bomb_spends"]
    return dict(credits=n,actions=actions,stage2_reaches=stage,stage2_rate=stage/n,stage2_wilson95=valid,
                stage2_entry_resources=[dict(seed=r["seed"],spares=r["stage2_spare_lives"],bombs=r["stage2_bombs"]) for r in rows if r["stage2_reached"]],
                means={k:float(np.mean([r[k] for r in rows])) for k in fields},
                medians={k:float(np.median([r[k] for r in rows])) for k in fields},
                moves=moves.tolist(),vertical_fraction=float(sum(moves[:3])+sum(moves[6:]))/actions,
                nonneutral_fraction=1-moves[4]/actions)

def compare(rows,reference,candidate):
    pairs=defaultdict(dict)
    for r in rows:pairs[r["seed"]][r["arm"]]=r
    result={};rng=np.random.default_rng(261126)
    for feature in ("stage2_reached","last_stage_native_tick","first_loss_native_tick","seconds","last_live_bombs","actual_bomb_spends"):
        per_lane=defaultdict(list)
        for pair in pairs.values():
            a,b=pair[reference],pair[candidate]
            if (a["host"],a["lane"],a["difficulty"])!=(b["host"],b["lane"],b["difficulty"]):raise ValueError("Broken same-host, same-lane pairing")
            per_lane[a["lane"]].append(float(b[feature])-float(a[feature]))
        samples=[]
        for differences in per_lane.values():
            x=np.asarray(differences);samples.append(x[rng.integers(len(x),size=(20000,len(x)))])
        all_values=np.concatenate(list(per_lane.values()))
        result[feature]=dict(pairs=len(all_values),candidate_minus_reference=float(all_values.mean()),wins=int((all_values>0).sum()),
            ties=int((all_values==0).sum()),losses=int((all_values<0).sum()),
            paired_lane_bootstrap95=np.quantile(np.concatenate(samples,axis=1).mean(1),[.025,.975]).tolist())
    return result

def report(study,mode):
    protocol=json.loads((study/"camera-protocol.json").read_bytes());native=study/"native17"
    reference=protocol["reference_arm"];candidate=protocol["candidate"]
    expected=set(protocol["eval_seeds" if mode=="held" else "train_seeds"]["17"])
    arms=set([reference,candidate] if mode=="held" else [reference])
    rows=[]
    for receipt in native.glob(mode+"-*/receipt.json"):
        r=json.loads(receipt.read_bytes())
        native_trace=native/receipt.parent.name/"native.tsv"
        if r["terminal_screen"] not in ("game_over","score_ranking") or r["terminal_spares"]!=-1 or not native_trace.is_file():
            raise ValueError("Original terminal incomplete")
        if r["difficulty"]!=int(protocol["rank_by_seed"]["17"][str(r["seed"])]) or r["continues"]!=0:raise ValueError("Original difficulty/Continue changed")
        r["actual_bomb_spends"]=sum(e["count"] for e in r["reward_events"] if e["kind"]=="spent_bombs")
        rows.append(r)
    if len(rows)!=len(expected)*len(arms) or {(r["seed"],r["arm"]) for r in rows}!={(s,a) for s in expected for a in arms}:
        raise ValueError("Registered original cohorts are incomplete")
    cells={}
    for rank in (2,3):
        cell=[r for r in rows if r["difficulty"]==rank]
        cells[rank]={"arms":{a:summarize([r for r in cell if r["arm"]==a]) for a in sorted(arms)}}
        if mode=="held":cells[rank]["paired"]=compare(cell,reference,candidate)
    published_fields=("seed","arm","host","lane","difficulty","actions","terminal_screen","maximum_stage","stage2_reached","stage2_spare_lives","stage2_bombs",
        "first_loss_native_tick","last_stage_native_tick","last_live_bombs","remaining_bombs","seconds","actual_bomb_spends","policy_sha256","trajectory_sha256","native_trace_sha256","terminal_observation_sha256")
    return dict(mode=mode,reference=reference,candidate=candidate if mode=="held" else None,cells=cells,
        scope="Full unassisted original TH17 trial credits. The seed controls 4-head policy sampling. The original game RNG is uncontrolled. Last playable Bomb stock excludes the post-death refill. Conditional stage-entry stocks are absent when no credit reaches Stage 2. Wilson intervals describe individual-arm stage arrival. Paired bootstrap intervals resample within fixed private game instance; zero-event bootstrap intervals do not bound population success probability.",
        credits=[{k:r[k]for k in published_fields} for r in sorted(rows,key=lambda r:(r["seed"],r["arm"]))])

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("study",type=Path);p.add_argument("mode",choices=["train","held"]);p.add_argument("output",type=Path)
    a=p.parse_args();o=report(a.study,a.mode);a.output.write_text(json.dumps(o,indent=2)+"\n");print("Complete original credits",len(o["credits"]))
