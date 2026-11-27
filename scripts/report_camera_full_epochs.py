"""Plot human movement and generalization through the four fixed full-epoch checkpoints."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.checkpoint_storage import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prospective',action='store_true')
    args=parser.parse_args();d=args.data.resolve();root=d/'four-title-complete-epochs';out=args.out.resolve();out.mkdir(exist_ok=True,parents=True)
    plan=json.loads((root/'protocol.json').read_bytes());epochs=plan['milestones'];cells={}
    protocol=digest(root/'protocol.json')
    models=[root/f'epoch-{e:02}' for e in epochs]
    files=[]
    if args.prospective:
        manifest=d/'unseen-later-stage-author-challenge/held-original-synchronized-frozen.json'
        population=json.loads(manifest.read_bytes())
        allocation=json.loads((manifest.parent/'prospective-six-players-frozen.json').read_bytes())
        expected={(v['sha256'],v['reserved_stage']) for v in allocation['sources']}
        results=population['qualified']+population['excluded']
        if len(results)!=6 or {(v['replay_sha256'],v['stage']) for v in results}!=expected:raise ValueError('Incomplete reserved roster')
        for v in population['qualified']:
            f=d/v['path']
            if digest(f)!=v['archive_sha256'] or v['split']!='held':raise ValueError('Changed prospective stage')
            files.append(f)
        if not files:
            (out/'excluded-roster.json').write_text(json.dumps(dict(source=digest(manifest),outcome='No synchronized held stage',exclusions=population['excluded']),indent=2)+'\n');return
        subprocess.run([sys.executable,'-m','scripts.compare_human_camera_policies','--models',*map(str,models),'--held',*map(str,files),'--out',str(out)],check=True)
        subprocess.run([sys.executable,'-m','scripts.score_human_bomb_onsets','--models',*map(str,models),'--held',*map(str,files),'--out',str(out/'onsets.json')],check=True)
        held=json.loads((out/'held.json').read_bytes())
        lookup={v['archive_sha256']:v for v in population['qualified']}
    for e,model in zip(epochs,models):
        report=json.loads((model/'report.json').read_bytes())
        if report['protocol']!=protocol or digest(model/'policy.ts')!=report['torchscript_sha256']:raise ValueError('Changed fit')
        rows=report['development']
        if args.prospective:
            rows={v['sha256']:dict(group=v['group'],title=lookup[v['sha256']]['title'],metrics=v['aligned']) for v in held[model.name]['replays'].values()}
        by_title={}
        for title in sorted({v['title'] for v in rows.values()}):
            players=defaultdict(list)
            for v in rows.values():
                if v['title']==title:players[v['group']].append(v['metrics'])
            metrics={}
            for k in ('all','change','start','turn','stop'):
                values=[];correct=n=0
                for stages in players.values():
                    total=sum(v[k]['n'] for v in stages)
                    if total:values.append(sum(v[k]['n']*v[k]['movement_nll'] for v in stages if v[k]['n'])/total)
                    correct+=sum(v[k]['correct'] for v in stages);n+=total
                metrics[k]=dict(mean_player_nll=float(np.mean(values)) if values else None,correct=correct,windows=n,players=len(values))
            by_title[title]=metrics
        cells[e]=dict(titles=by_title,weighted_joint_training_loss=report['history'][-1]['mean_weighted_joint_loss'],actor=report['torchscript_sha256'])
    (out/'by-title-and-epoch.json').write_text(json.dumps(dict(protocol=protocol,scope='prospective' if args.prospective else 'repeated development',cells=cells),indent=2,allow_nan=False)+'\n',encoding='utf8')
    fig,axes=plt.subplots(1,3,figsize=(11.6,3.9),layout='constrained')
    labels={'all':'All movement','start':'Start moving','turn':'Turn while moving'}
    for ax,(k,label) in zip(axes,labels.items()):
        for title in cells[epochs[0]]['titles']:
            ax.plot(epochs,[cells[e]['titles'][title][k]['mean_player_nll'] for e in epochs],marker='o',label=title)
        ax.set_title(label);ax.set_xlabel('Completed full training epochs');ax.set_xticks(epochs);ax.grid(alpha=.22)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Negative log likelihood per player');axes[-1].legend(frameon=False)
    fig.suptitle('Four-title human SFT on '+('prospectively held players' if args.prospective else 'repeated development players'))
    fig.savefig(out/'movement-by-epoch.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
