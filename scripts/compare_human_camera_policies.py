"""Compare frozen camera policies on the same held players and movement transitions."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.train_human_camera_policy import evaluation, load_replay
from scripts.train_replay_camera_encoder import save_json, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',type=Path,required=True)
    p.add_argument('--held',nargs='+',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    rows=[load_replay(path) for path in a.held]
    groups={row['group'] for row in rows}
    results={}
    with torch.inference_mode():
        for directory in a.models:
            report=json.loads((directory/'report.json').read_bytes())
            if groups.intersection(x['group'] for x in report['train']):
                raise ValueError("Training player in held set")
            model_path=directory/'policy.ts'
            if sha(model_path)!=report['torchscript_sha256']:raise ValueError("Frozen artifact has changed")
            model=torch.jit.load(str(model_path),map_location=device).eval()
            metrics={}
            for row in rows:
                calls=[model(b.to(device).float()/255) for b in torch.from_numpy(row['raw']).split(64)]
                probs=[torch.cat(outputs).softmax(1).cpu().numpy() for outputs in zip(*calls)]
                matched=evaluation(row,probs)
                shifted=evaluation(row,[np.roll(x,len(x)//2,axis=0) for x in probs])
                metrics[row['name']]={'group':row['group'],'aligned':matched,'shifted':shifted,'sha256':row['sha256']}
            results[directory.name]={'checkpoint':report['torchscript_sha256'],'replays':metrics}
    a.out.mkdir(parents=True,exist_ok=True)
    save_json(a.out/'held.json',results)
    conditions=['all','change','start','turn']
    fig,axes=plt.subplots(1,len(results),squeeze=False,figsize=(5*len(results),3.5),layout='constrained',sharey=True)
    for ax,(name,results_model) in zip(axes.flat,results.items()):
        for i,mode in enumerate(('aligned','shifted')):
            count=[sum(r[mode][cond]['n'] for r in results_model['replays'].values()) for cond in conditions]
            hit=[sum(r[mode][cond]['correct'] for r in results_model['replays'].values()) for cond in conditions]
            acc=np.divide(hit,count,out=np.zeros(4),where=np.array(count)>0)
            bars=ax.bar(np.arange(4)+(i-.5)*.38,acc*100,.37,label=mode,color=('#13766f','#a4b5c1')[i])
            ax.bar_label(bars,fmt='%.0f',fontsize=8,padding=2)
        ax.set_title(name);ax.set_xticks(range(4),[f'{s}\n(n={n:,})' for s,n in zip(conditions,count)])
        ax.set_ylim(0,100);ax.legend(frameon=False);ax.grid(axis='y',alpha=.18)
    axes[0,0].set_ylabel('Expert movement predicted correctly (%)')
    fig.suptitle(f'Original replay cameras · {len(groups)} held player(s), {len(rows)} stage(s)')
    fig.savefig(a.out/'held-camera.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':main()
