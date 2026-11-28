"""Draw per-seed original native exposure and pre-loss Bomb counts from scored TH08 credits."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def make(report,output):
    data=json.loads(report.read_bytes())
    arms=["lambda1-parent","lambda1-continuation"]
    colors=["#315c8c","#b65819"]
    records=data["by_original"]
    pairs=sorted({r["seed"] for r in records})
    if not pairs:raise ValueError("No native endpoints to plot")
    for row in records:
        if row["terminal_stage"]!=1 or row.get("stage1_at_risk_ticks") is None:
            raise ValueError("Cross-stage or unknown endpoint requires an explicit Stage 1 exposure review")
    fig,(ticks,bombs)=plt.subplots(1,2,figsize=(10.2,5.0),gridspec_kw={"width_ratios":[2.45,1]},sharey=True)
    largest=max(r["stage1_at_risk_ticks"] for r in records)
    for index,seed in enumerate(pairs):
        physical=[r for r in records if r["seed"]==seed]
        lines=[("stage1_at_risk_ticks",ticks),("losses_with_positive_pre_loss_bomb",bombs)]
        for key,ax in lines:
            points=[(r[key],index+(-.13 if r["arm"]==arms[0] else .13)) for r in physical]
            if len(points)==2:ax.plot(*zip(*points),color="#bcc4ca",linewidth=1.2,zorder=1)
            for r,(x,y) in zip(physical,points):
                arm=arms.index(r["arm"])
                ax.scatter(x,y,s=44,marker="o" if arm==0 else "D",linewidths=1.5,edgecolor=colors[arm],
                           facecolor=colors[arm] if r["strict_control"] else "white",zorder=2)
                if ax is ticks:
                    ax.annotate(f"{x:,}",(x,y),xytext=(7,2 if arm else -2),textcoords="offset points",
                                va="center",color=colors[arm],fontsize=8)
    labels=[str(s) for s in pairs]
    ticks.set_yticks(list(range(len(pairs))),labels)
    ticks.set_ylim(len(pairs)-.6,-.5)
    ticks.set_xlim(0,largest*1.17)
    ticks.set_xlabel("Native Stage 1 ticks at risk")
    ticks.set_ylabel("Paired controller seed")
    bombs.set_xlim(-.4,3.5)
    bombs.set_xticks([0,1,2,3])
    bombs.set_xlabel("Deaths holding ≥1 Bomb")
    for ax in (ticks,bombs):
        ax.spines[["top","right","left"]].set_visible(False)
        ax.tick_params(axis="y",length=0)
        ax.grid(axis="x",alpha=.22)
    handles=[Line2D([],[],marker=m,color=c,linestyle="None",label=l,markersize=7) for m,c,l in
             zip(("o","D"),colors,("Parent PPO","PPO after 29 additional stocks"))]
    fig.legend(handles=handles,loc="upper left",bbox_to_anchor=(.09,.87),ncol=2,frameon=False)
    fig.suptitle("TH08 Normal · Ghost Team",x=.09,ha="left",y=.985,fontsize=17)
    fig.text(.09,.891,"Paired full-stock original games, four learned controls and no continues",fontsize=10,color="#43515a")
    fig.text(.09,.039,f'{data["physical_credits"]} original endpoints   ·   {data["strict_exposure_pairs"]} strict pairs   ·   '
                         'Filled markers pass timing and input checks. Hollow markers indicate a technical failure.',
             fontsize=8,color="#43515a")
    fig.subplots_adjust(left=.13,right=.97,top=.73,bottom=.17,wspace=.25)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("report",type=Path);parser.add_argument("figure",type=Path)
    args=parser.parse_args();make(args.report,args.figure)
