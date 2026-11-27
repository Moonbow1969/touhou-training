"""Compare sampled Bomb presses with original inventory changes and stock at each death."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean

import numpy as np


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source,"sha256").hexdigest()


def audit(path,receipt):
    item=json.loads(receipt.read_bytes())
    if not item["physical_valid"]:return None
    source=path/item["recovered_source"];packed=path/item["record"]["file"]
    actual=json.loads((source/"receipt.json").read_bytes())
    timing=source/"timing.jsonl"
    if digest(packed)!=item["record"]["sha256"] or digest(timing)!=actual["timing_sha256"]:
        raise ValueError(f"Physical source differs for {receipt}")
    moments=[json.loads(line) for line in timing.read_bytes().splitlines()]
    play=[row for row in moments if row["phase"]=="policy"]
    with np.load(packed,allow_pickle=False) as q:
        pressed=q["action"][:,3].astype(bool)
    if len(pressed)!=len(play) or moments[-1]["phase"]!="terminal":
        raise ValueError(f"Incomplete physical credit {receipt}")
    rises=pressed & ~np.r_[False,pressed[:-1]]
    losses=[j for j,v in item["audit"]["life_events"] if v<0]
    stocks=[play[j]["bombs"] for j in losses]+[item["audit"]["last_playable_bombs"]]
    decreases=[(j,-v) for j,v in item["audit"]["bomb_events"] if v<0 and all(abs(j-loss)>6 for loss in losses)]
    return dict(seed=item["seed"],rank=item["rank"],arm=item["arm"],team=item["team"],receipt_sha256=digest(receipt),
                decisions=len(pressed),bomb_pressed=int(pressed.sum()),bomb_rising_edges=int(rises.sum()),
                same_life_hud_decreases=len(decreases),same_life_bomb_units=sum(v for _,v in decreases),
                decreases_with_x_onset_in_prior_four_decisions=sum(bool(rises[max(0,j-3):j+1].any()) for j,_ in decreases),
                observed_stock_before_each_death=stocks)


def report(native,role):
    rows=[];cells=defaultdict(list)
    src=native/("training-receipts" if role=="train" else "receipts")
    all_files=list(src.glob("*.json"))
    for f in sorted(all_files):
        row=audit(native,f)
        if row:
            rows.append(row);cells[row["team"],row["rank"],row["arm"]].append(row)
    groups=[]
    for (team,rank,arm),values in sorted(cells.items()):
        death_stock=[x for r in values for x in r["observed_stock_before_each_death"] if x is not None]
        counts={k:sum(v[k] for v in values) for k in ("decisions","bomb_pressed","bomb_rising_edges","same_life_hud_decreases","same_life_bomb_units","decreases_with_x_onset_in_prior_four_decisions")}
        groups.append(dict(team=team,rank=rank,arm=arm,credits=len(values),counts=counts,death_stock_readings=len(death_stock),
                           deaths_with_positive_bombs=sum(x>0 for x in death_stock),death_stock_mean=mean(death_stock) if death_stock else None))
    return dict(original_receipts=len(all_files),qualified=len(rows),cells=groups,credits=rows,
                interpretation="One X onset is one transition from released to pressed in the host-applied key stream. An on-screen Bomb inventory decrement is a two-observation-confirmed decrease more than six decisions from a spare-life change, using the same event rule as training. It can cost multiple Bomb units. Actual casts without a readable stock change are unobserved. Before each verified death use the last sampled pre-change HUD stock, and for the final life use the last playable stock before Continue. The held figures describe these players and establish no causal effect of a reward coefficient.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--native",type=Path,required=True)
    p.add_argument("--role",choices=("train","held"),required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.write_text(json.dumps(report(a.native,a.role),indent=2)+"\n",encoding="utf8")
