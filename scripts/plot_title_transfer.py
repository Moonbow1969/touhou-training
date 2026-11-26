"""Plot fixed-budget multi-title action prediction and its direction-change diagnostic."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    def load(name):
        return json.loads((args.directory/name).read_bytes())
    matrix=load("transfer-matrix-live.json")
    matched=load("switches.json")["scores"]
    weighted=load("switches25.json")["scores"]
    titles=["13","14","17"]
    colors=["#003f5c","#d15a31"]
    fig,axes=plt.subplots(1,3,figsize=(11.5,3.9),layout="constrained")
    for i,(ax,metric) in enumerate(zip(axes,["move_nll","starts","turns"])):
        for j,color in enumerate(colors):
            values=[]
            for t in titles:
                labels=[f"seed-{s}-all" for s in matrix["seeds"]]
                if i==0:
                    labels=labels if j==0 else [f"seed-{s}-without-{t}" for s in matrix["seeds"]]
                    values.append([a["scores"][t][metric] for a in matrix["arms"] if a["label"] in labels])
                else:
                    scores=matched if j==0 else weighted
                    values.append([100*scores[k][t][metric]["accuracy"] for k in labels])
            x=np.arange(3)+(j-.5)*.34
            labels=("Joint","Title excluded") if i==0 else ("Uniform","25% switch draws")
            ax.bar(x,np.mean(values,axis=1),width=.32,color=color,alpha=.85,label=labels[j])
            for center,points in zip(x,values):
                ax.scatter([center-.035,center+.035],points,s=13,c="black",zorder=3,linewidth=0)
        ax.set_xticks(np.arange(3),[f"TH{t}" for t in titles])
        ax.set_title(["Movement NLL ↓","Correct direction at start ↑","Correct direction at turn ↑"][i],fontsize=12)
        ax.set_ylabel("Nats per decision" if i==0 else "Accuracy (%)")
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(frameon=False,fontsize=8,loc="upper left")
        ax.margins(y=.32)
    fig.suptitle("Four held human replays  ·  two optimization seeds  ·  2,400 steps per fit",fontsize=14)
    fig.savefig(args.directory/"title-transfer.png",dpi=160)


if __name__=="__main__":
    main()
