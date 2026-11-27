"""Visualize matched broad versus high-rank replay SFT credits in TH10 and TH13."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(study):
    th13=json.loads((study/"held-th13/evaluation/paired-by-rank.json").read_bytes())
    arms=("broad38","ranked3")
    fig,axes=plt.subplots(2,2,figsize=(8.9,7.2),layout="constrained")
    for col,(level,rank) in enumerate([("Hard",2),("Lunatic",3)]):
        th10=json.loads((study/f"held-th10/native10/{level.lower()}/paired.json").read_bytes())
        score={arm:{r["seed"]:r["raw_score"]/1000 for r in th10["credits"]["held-"+arm]} for arm in arms}
        th10_stage=[th10["group"]["held-"+arm]["stage2_reached"] for arm in arms]
        th13_stage=[th13[str(rank)]["groups"][arm]["stage2"] for arm in arms]
        clock=th13[str(rank)]["primary"]["per_seed"]
        for i,(table,stage,title,ylabel) in enumerate((
            (score,th10_stage,"TH10","Original score (thousands)"),
            (clock,th13_stage,"TH13","Last observed Stage 1 script tick"))):
            assert set(table[arms[0]])==set(table[arms[1]]) and len(table[arms[0]])==12
            ax=axes[i,col]
            for j,seed in enumerate(sorted(table[arms[0]])):
                x=np.array([0.,1.])+(j-5.5)*.009
                y=[table[arm][seed] for arm in arms]
                ax.plot(x,y,color="#b4bcc4",lw=1,alpha=.85,zorder=1)
                ax.scatter(x,y,c=["#31678c","#a4532e"],s=18,zorder=2)
            ax.scatter([0,1],[np.mean(list(table[arm].values())) for arm in arms],color="#15202b",
                       marker="_",s=330,linewidths=2.3,zorder=3)
            ax.set_xticks([0,1],["Broad 38-player","Rank-specialist"])
            ax.set_ylabel(ylabel);ax.set_xlim(-.20,1.20);ax.grid(axis="y",alpha=.2)
            ax.set_title(f"{title} {level} · Stage 2 {stage[0]}/12 → {stage[1]}/12",loc="left",fontsize=11)
    fig.suptitle("Human replay policies on complete original-game credits",fontweight="bold")
    path=study/"paired-original-ranks.png";fig.savefig(path,dpi=170);plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("study",type=Path)
    plot(parser.parse_args().study)
