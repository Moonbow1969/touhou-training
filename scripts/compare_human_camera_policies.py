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
    p.add_argument('--model-names',nargs='+',help='Unique display names in the same order as the model directories.')
    p.add_argument('--held',nargs='+',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sft-parent',type=Path,help='Supervised parent directory for a first-generation PPO checkpoint')
    p.add_argument('--difference-ticks',choices=(4,8),type=int,default=4)
    a=p.parse_args()
    names=a.model_names or [directory.name for directory in a.models]
    if len(names)!=len(a.models) or len(set(names))!=len(names):
        p.error('Supply one unique name per checkpoint')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    rows=[load_replay(path,include_eight_tick_camera=a.difference_ticks==8) for path in a.held]
    if a.difference_ticks==8:
        for row in rows:
            good=row["valid8"]
            row["raw"]=row["raw8"][good]
            row["action"]=row["action"][good];row["prior"]=row["prior"][good]
    groups={row['group'] for row in rows}
    results={}
    with torch.inference_mode():
        for name,directory in zip(names,a.models):
            report=json.loads((directory/'report.json').read_bytes())
            if 'train' in report:
                provenance=report['train'];expected=report['torchscript_sha256']
            else:
                if a.sft_parent is None:raise ValueError('Missing SFT parent for PPO provenance')
                source=json.loads((a.sft_parent/'report.json').read_bytes())
                if (report['parent_sft_sha256']!=source['sft'] or
                    report['native_behavior_sha256']!=source['torchscript_sha256']):
                    raise ValueError('PPO does not descend directly from this supervised checkpoint')
                provenance=source['train'];expected=report['policy_sha256']
            if groups.intersection(x['group'] for x in provenance):
                raise ValueError("Training player in held set")
            model_path=directory/'policy.ts'
            if sha(model_path)!=expected:raise ValueError("Frozen artifact has changed")
            model=torch.jit.load(str(model_path),map_location=device).eval()
            metrics={}
            for row in rows:
                calls=[model(b.to(device).float()/255) for b in torch.from_numpy(row['raw']).split(64)]
                probs=[torch.cat(outputs).softmax(1).cpu().numpy() for outputs in list(zip(*calls))[:4]]
                matched=evaluation(row,probs)
                shifted=evaluation(row,[np.roll(x,len(x)//2,axis=0) for x in probs])
                metrics[row['name']]={'group':row['group'],'aligned':matched,'shifted':shifted,'sha256':row['sha256']}
            results[name]={'checkpoint':expected,'replays':metrics}
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
    fig.suptitle(f'Original replay cameras · {a.difference_ticks}-tick history · {len(groups)} held player(s), {len(rows)} stage(s)')
    fig.savefig(a.out/'held-camera.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':main()
