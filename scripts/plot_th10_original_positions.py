"""Plot the observed player positions for each arm of a matched full-credit TH10 cohort."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.collect_th10_full_action import save


def main(base):
    groups={};density={};edges_x=np.linspace(-192,192,25);edges_y=np.linspace(0,448,29)
    for arm in ('ppo3','camera_sft','camera_ppo'):
        records=[];maps=[]
        for p in sorted((base/('held-'+arm)).glob('*/report.json')):
            r=json.loads(p.read_bytes())
            with np.load(p.parent/'behavior.npz') as b:start=int(b['action_utc_ms'][0])
            with gzip.open(p.parent/'native.csv.gz','rt') as f:
                rows=[s for s in csv.DictReader(f) if int(s['utc_ms'])>=start and int(s['lives'])>=0]
            xy=np.asarray([[float(s['x']),float(s['y'])] for s in rows]);x,y=xy.T
            valid=(x>=-192)&(x<=192)&(y>=0)&(y<=448);x,y=x[valid],y[valid]
            cells=np.histogram2d(y,x,bins=[edges_y,edges_x])[0]
            maps.append(cells/cells.sum())
            records.append(dict(seed=r['seed'],in_play_samples=len(x),outside_grid=int((~valid).sum()),
                lower_quarter_fraction=float(np.mean(y>=336)),outer_x_quarters_fraction=float(np.mean(np.abs(x)>=144)),
                lower_corner_fraction=float(np.mean((y>=336)&(np.abs(x)>=144)))))
        assert [r['seed'] for r in records]==list(range(754001,754007))
        density[arm]=np.mean(maps,axis=0)*100
        groups[arm]={'credits':records,**{k:float(np.mean([r[k] for r in records])) for k in
            ('lower_quarter_fraction','outer_x_quarters_fraction','lower_corner_fraction')}}
    names=['Portable PPO3','Human camera SFT','Camera PPO1']
    fig,axes=plt.subplots(1,3,figsize=(9,4.1),layout='constrained',sharey=True)
    vmax=max(v.max() for v in density.values())
    for ax,name,(arm,values) in zip(axes,names,density.items()):
        im=ax.imshow(values,origin='upper',extent=(-192,192,448,0),vmin=0,vmax=vmax,cmap='magma',interpolation='nearest')
        ax.set_title(name);ax.set_xlabel('Original player x');ax.set_xticks([-192,0,192])
    axes[0].set_ylabel('Original player y');axes[0].set_yticks([0,112,224,336,448])
    fig.colorbar(im,ax=axes,label='Mean credit time per 16 × 16 cell (%)',shrink=.8)
    fig.suptitle('Where each policy moved in TH10 Normal')
    fig.supxlabel('Six matched original-game credits per policy. Each credit contributes equal weight.',fontsize=9)
    fig.savefig(base/'native-position-occupancy.png',dpi=180);plt.close(fig)
    save(base/'native-position-occupancy.json',dict(coordinates='Playfield in original units x [-192,192], y [0,448]. Native position sensors surveyed during control through last life; respawn observations retained. Each credit is normalized before averaging.',groups=groups))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('base',type=Path)
    main(p.parse_args().base)
