"""Render frozen held results and verify every learned artifact for the public archive."""
from pathlib import Path
import hashlib,json,shutil,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def sha(p):
    with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()

def render(study):
    fit=study/'fit';d=json.loads((fit/'result.json').read_bytes());p=json.loads((study/'protocol.json').read_bytes())
    assert d['protocol_sha256']==sha(study/'protocol.json') and d['trainer_sha256']==sha(study/'fit.py')
    traces=[];table=[];unique={}
    for seed in p['seeds']:
        with np.load(fit/f'{seed}-batches.npz')as data:
            matrices={arm:data[arm]for arm in p['arms']}
            for x,y in zip(*matrices.values()):assert len(set(x)&set(y))==16
            unique[seed]={}
            for arm,matrix in matrices.items():
                assert matrix.shape==(536,32)
                folder=fit/f'{seed}-{arm}'
                details=json.loads((folder/'fit.json').read_bytes())
                assert sha(folder/'policy.ts')==details['policy_sha256'] and sha(folder/'conditioned.pt')==details['train_state_sha256']
                assert details['sampling_sha256']==hashlib.sha256(matrix.astype('<i8').tobytes()).hexdigest()
                assert details['export_max_abs']<1e-5 and len(details['progress'])==8
                unique[seed][arm]=int(len(np.unique(matrix)))
                traces.append(dict(seed=seed,arm=arm,progress=details['progress']))
    summary={}
    for step in (1,2):
        key=f'zylo-reimua-s{step}';summary[key]={}
        for arm in p['arms']:
            rows=[r[arm]['held'][key] for r in d['results'].values()]
            row={f:np.mean([r[f]for r in rows],axis=0).tolist() for f in
                 ('accuracy','nll','switch_move_acc','stable_move_acc','switch_move_nll','stable_move_nll','locator_median_native_px','mean_bomb_probability')}
            row['bomb_positives']=rows[0]['bomb_positive'];row['bomb_argmax_tp_per_seed']=[r['bomb_true_positive']for r in rows]
            summary[key][arm]=row
    export=dict(primary_passed=d['passed'],stages=summary,differences=d['stages'],unique_rows_used_per_arm_per_seed=unique,trained_actors=6,
                optimizer_steps_total=3216,training_sample_presentations=102912,device=d['gpu'],gpu_peak_allocated_mb=d['gpu_peak_allocated_mb'])
    (study/'summary.json').write_text(json.dumps(export,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(11,3.55),layout='constrained')
    colors={'reimua':'#175a72','mixed':'#b45927'}
    labels={'reimua':'Reimu A','mixed':'Cross-shot'}
    for arm in p['arms']:
        series=np.array([[v['action_ce']for v in t['progress']]for t in traces if t['arm']==arm])
        x=np.arange(67,537,67)
        axes[0].plot(x,series.mean(0),color=colors[arm],label=labels[arm])
        axes[0].fill_between(x,series.min(0),series.max(0),alpha=.14,color=colors[arm])
    axes[0].set(title='Training action loss',xlabel='Optimizer steps',ylabel='Four-head cross-entropy')
    axes[0].legend(frameon=False)
    for ax,k,title in zip(axes[1:],('move','switch'),('Held movement','Held direction changes')):
        for j,stage in enumerate((1,2),1):
            vals=d['stages'][f'zylo-reimua-s{stage}']['paired']
            ax.scatter(np.full(3,j)+np.array([-.05,0,.05]),[100*v[k]for v in vals],s=35,color='#175a72')
            avg=np.mean([100*v[k]for v in vals]);ax.hlines(avg,j-.17,j+.17,color='#b45927',lw=2)
        ax.axhline(0,color='#444444',lw=.7)
        ax.axhline(1.5 if k=='move' else -.5,color='#888888',ls=':',lw=.8)
        ax.set(title=title,xticks=[1,2],xticklabels=['Stage 1','Stage 2'],xlim=[.5,2.5],ylabel='Cross-shot minus Reimu A (pp)')
    fig.savefig(study/'matched-sft.png',dpi=160)
    fig.savefig(study/'matched-sft.svg')
if __name__=='__main__':
    import sys
    render(Path(sys.argv[1]))
