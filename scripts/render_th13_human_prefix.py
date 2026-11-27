"""Render the clock-aligned TH13 human prefix and live neural handoff."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def draw(root):
    study=root/"engineering/th13-live-midstage-handoff";live=study/"live-1271303/engineering-humanprefix-visual-1271303"
    with (live/"prefix.csv").open() as f:prefix=list(csv.DictReader(f))
    data=np.fromfile(live/"steps.bin","<f4").reshape(-1,143)
    with np.load(root/"human-trial13/train/th13trial-th12_ud003550-s1.npz") as a, np.load(live/"rollout.npz") as b:
        historical=a["obs"][np.flatnonzero(a["native_frame"]==3000)[0]].copy();actual=b["frames"][0].copy()
    tick=[int(r["stage_tick"]) for r in prefix]+data[:,135].astype(int).tolist()
    stock=[max(-1,int(r["spares"])) for r in prefix]+list(data[:,128])
    bombs=[int(r["bombs"]) for r in prefix]+list(data[:,130])
    fig,axes=plt.subplots(1,3,figsize=(11.4,3.6),width_ratios=[1,1,2.5],layout="constrained")
    for ax,arr,title in zip(axes[:2],(historical,actual),("Human replay","Live policy input")):
        ax.imshow(arr,cmap="gray",vmin=0,vmax=255,interpolation="nearest");ax.axis("off");ax.set_title(title,fontsize=12)
    ax=axes[2];ax.step(tick,stock,where="post",label="Spare lives",color="#247a64",linewidth=2.2)
    ax.step(tick,bombs,where="post",label="Bombs",color="#da8144",linewidth=1.8)
    ax.axvline(3000,color="#33374a",linewidth=1,linestyle="--")
    ax.text(1500,2.25,"Human input",ha="center");ax.text(4010,2.25,"Visual PPO",ha="center")
    ax.set(yticks=[-1,0,1,2],yticklabels=["Game Over","0","1","2"],ylim=(-1.2,2.6),xlabel="Original Stage 1 tick",xlim=(0,5050))
    ax.set_xticks([0,1000,2000,3000,4000,5025]);ax.grid(axis="y",alpha=.17)
    ax.spines[["top","right"]].set_visible(False);ax.legend(loc="lower left",ncol=2,frameon=False)
    axes[0].text(.5,-.05,"Tick 3,000",transform=axes[0].transAxes,ha="center")
    axes[1].text(.5,-.05,"Tick 3,000",transform=axes[1].transAxes,ha="center")
    fig.suptitle("TH13 Hard   ·   Reimu   ·   Human-input handoff to the four-game PPO",fontsize=14)
    fig.savefig(study/"original-handoff.png",dpi=160);plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);draw(p.parse_args().root)
