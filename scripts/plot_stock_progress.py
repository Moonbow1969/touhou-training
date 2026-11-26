"""Plot observed Stage 2 entries and entry resources across frozen original-game policy evaluations."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.summarize_stock_story_cohort import interval


def rows_from_story(path):
    report=json.loads(path.read_text(encoding="utf-8-sig"))
    task=report["specification"]["task"]
    if (report["status"]!="complete" or task["title"]!="TH14" or task["difficulty"]!="Normal"
            or task["shot"]!="Reimu A" or not task["all_native_lives"]):
        raise ValueError(f"Unfinished or incompatible credit report: {path.name}")
    for item in report["completed"]:
        result=item["result"]
        if result["native_continues"] or result["outcome"] not in ("game_over","one_credit_clear","target_reached"):
            raise ValueError("Unverified terminal credit.")
        stage2=next((s for s in result["stage_records"] if s["stage"]==2),None)
        resources=stage2["entry"]["resources"] if stage2 else {}
        yield (bool(stage2),resources.get("lives"),resources.get("bombs"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original",type=Path,required=True,help="Published SFT, PPO211 and PPO397 full-credit CSV.")
    parser.add_argument("--cohort",nargs=2,action="append",metavar=("LABEL","REPORT"),required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    original={"sft":"Original SFT","ppo211":"Original PPO-211","ppo397":"Original PPO-397"}
    groups={v:[] for v in original.values()}
    with args.original.open(encoding="utf-8-sig",newline="") as stream:
        for line in csv.DictReader(stream):
            if line["arm"] in original:
                if line["stage2_arrival"] not in ("True","False"):
                    raise ValueError("Missing verified original Stage 2 indicator.")
                reached=line["stage2_arrival"]=="True"
                groups[original[line["arm"]]].append((reached,int(line["stage2_lives"] or 0) if reached else None,
                                                      int(line["stage2_bombs"] or 0) if reached else None))
    for label,path in args.cohort:
        groups.setdefault(label,[]).extend(rows_from_story(Path(path)))
    if not all(groups.values()):
        raise ValueError("Each displayed policy needs at least one completed native credit.")
    figure,(rates,stocks)=plt.subplots(1,2,figsize=(10.6,1.85+.5*len(groups)),sharey=True,
                                     gridspec_kw=dict(width_ratios=[1.9,1.1]))
    for i,(label,credits) in enumerate(groups.items()):
        success=[r for r in credits if r[0]]
        n,k=len(credits),len(success)
        low,high=interval(k,n)
        color="#476e98" if label in original.values() else "#177753"
        y=len(groups)-i-1
        rates.plot([low,high],[y,y],color=color,lw=2)
        rates.scatter(k/n,y,color=color,s=37,zorder=3)
        rates.text(1.07,y,f"{k}/{n}",va="center",fontsize=10)
        text=" · ".join(f"{r[1]}/{r[2]}" for r in success) or "—"
        stocks.text(.1,y,text,va="center",fontsize=11)
    rates.set(yticks=np.arange(len(groups)),yticklabels=list(groups)[::-1],xlim=(-.045,1.22),
              xticks=[0,.25,.5,.75,1],xticklabels=["0","25","50","75","100"],
              xlabel="Observed Stage 2 entry rate (%)")
    rates.set_title("Reached Stage 2",loc="left",fontsize=12)
    stocks.set_title("Spare lives / Bombs at entry",loc="left",fontsize=12)
    rates.grid(axis="x",alpha=.2)
    for ax in (rates,stocks):
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(axis="y",length=0)
    stocks.set_xlim(0,1);stocks.set_xticks([]);stocks.spines["bottom"].set_visible(False)
    figure.suptitle("Stock TH14 Normal Reimu A  ·  full-life, zero-Continue credits",x=.025,ha="left",fontsize=15)
    figure.text(.025,.025,"Bars show 95% Wilson intervals. Each stock pair is one arrival. Native RNG is uncontrolled.",fontsize=9)
    figure.tight_layout(rect=[0,.065,1,.95])
    figure.savefig(args.out,dpi=160)
    plt.close(figure)

if __name__=="__main__":
    main()
