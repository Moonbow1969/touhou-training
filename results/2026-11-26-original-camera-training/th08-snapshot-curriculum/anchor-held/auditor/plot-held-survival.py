"""Plot the registered independent held first-loss curves and each physical outcome."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def render(outcomes, report, target):
    physical=json.loads(outcomes.read_bytes())["attempts"]
    data=json.loads(report.read_bytes())
    result=data["primary_qualified"]
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),gridspec_kw={"width_ratios":[1.4,1]})
    c={"lambda1-parent":"#326584","mixed-ppo":"#a34526"}
    names={"lambda1-parent":"Frozen parent","mixed-ppo":"Curriculum PPO"}
    for j,(arm,color) in enumerate(c.items()):
        metrics=result["arms"][arm]
        points=metrics["kaplan_meier"]
        times=[0]+[p["tick"] for p in points]
        values=[1]+[p["survival"] for p in points]
        axes[0].step(times,values,where="post",label=f"{names[arm]} (n={metrics['n']})",color=color,linewidth=2)
        censored=[p for p in points if p["right_censored"]]
        if censored:
            axes[0].scatter([p["tick"] for p in censored],[p["survival"] for p in censored],color=color,marker="+")
        pts=[r for r in physical if r["arm"]==arm and r["physical_endpoint_verified"]]
        for row in pts:
            marker="x" if not row["qualified"] else (">" if row["right_censored"] else "o")
            axes[1].scatter(row["ticks"],row["time_block"]+(j-.5)*.25,s=20,color=color,marker=marker,alpha=.84)
    axes[0].set(xlabel="Original ticks after the common Stage 1 state",ylabel="Share still holding both initial spares",ylim=(-.02,1.02))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    axes[0].legend(frameon=False,fontsize=8)
    axes[0].set_title("First spare loss",loc="left",fontsize=12)
    axes[1].set(xlabel="Original ticks after restore",ylabel="Chronological block",yticks=[1,5,10,15,20])
    axes[1].invert_yaxis()
    axes[1].set_title("Each observed original",loc="left",fontsize=12)
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(alpha=.13)
        ax.tick_params(labelsize=8)
    fig.suptitle("TH08 Normal Ghost from Stage 1 tick 2,201",x=.09,ha="left",fontsize=15,fontweight="bold")
    fig.text(.09,.02,"The arms use distinct controller seeds. A cross marks a native endpoint outside the control gate. A triangle marks a right-censored credit.",fontsize=8)
    fig.tight_layout(rect=(0,.06,1,.93))
    fig.savefig(target,dpi=170,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("outcomes",type=Path)
    parser.add_argument("report",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    render(args.outcomes,args.report,args.output)
