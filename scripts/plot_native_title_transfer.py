"""Show each native TH10 held credit, its matched controller seed and its censoring limit."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    panels=[("zero-shot","Random init → human SFT","random","sft"),
            ("warm-ppo-held","Human SFT → SFT + PPO","sft","ppo"),
            ("motion-held","SFT input ablation","sft","alternative_sft")]
    fig,axes=plt.subplots(1,3,figsize=(10.6,3.8),layout="constrained")
    for ax,(folder,title,left,right) in zip(axes,panels):
        journal=args.directory/folder/"report.json"
        rows=json.loads(journal.read_bytes())["credits"] if journal.is_file() else []
        seeds=sorted({r["seed"] for r in rows})
        for i,seed in enumerate(seeds):
            pair={r["arm"]:r for r in rows if r["seed"]==seed}
            if {left,right}!=set(pair):
                continue
            offset=(i-(len(seeds)-1)/2)*.035
            x=[offset,1+offset]
            vals=[pair[k]["decisions"] for k in (left,right)]
            ax.plot(x,vals,color="#889a9b",lw=1)
            for pos,key,value in zip(x,(left,right),vals):
                ax.scatter(pos,value,marker="^" if pair[key]["outcome"]=="horizon" else "o",
                           c="#056574" if key==left else "#cc6029",s=48,zorder=3)
        labels={"random":"Random init","sft":"Human SFT","ppo":"SFT + PPO","alternative_sft":"SFT + motion"}
        ax.set_xticks([0,1],[labels[left],labels[right]])
        ax.set_title(title,fontsize=11)
        ax.set_ylim(0,245)
        ax.set_xlim(-.32,1.32)
        ax.axhline(220,ls="--",c="#bbbbbb",lw=1)
        ax.spines[["right","top"]].set_visible(False)
    axes[0].set_ylabel("Confirmed decisions to first miss")
    fig.suptitle("TH10 Normal trial  ·  paired controller seeds  ·  triangles reached the 220-decision limit",fontsize=12)
    fig.savefig(args.out,dpi=170)


if __name__=="__main__":
    main()
