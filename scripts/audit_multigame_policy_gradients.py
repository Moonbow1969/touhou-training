"""Measure initial on-policy gradient agreement on native TRAIN credits across title and difficulty."""
import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scripts.checkpoint_storage import digest
from scripts.train_full_resolution_camera_ppo import CameraActorCritic,CameraPolicy,chosen,credit


def cosine(a,b):
    return float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-12))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--study',choices=('five-game-online','five-game-generation2'),default='five-game-online')
    args=p.parse_args();d=args.data.resolve()
    first=args.study=='five-game-online'
    root=d/args.study;manifest=root/('training/credits.json' if first else 'corrected/credits.json')
    plan=json.loads((root/'camera-protocol.json').read_bytes())
    report=json.loads((root/'policy/report.json').read_bytes())
    if digest(manifest)!=report['manifest_sha256']:raise ValueError('Changed training population')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    parent=d/('specialist-vs-shared/merged-equal' if first else 'five-game-online/policy')
    if digest(parent/'policy.ts')!=plan['behavior_sha256']:raise ValueError('Incorrect parent')
    camera=CameraPolicy()
    state=torch.load(parent/('camera-policy.pt' if first else 'camera-ppo.pt'),weights_only=True,map_location='cpu')
    if any('policy.spatial_context.' in n for n in state):camera.add_spatial_context()
    model=CameraActorCritic(camera)
    if first:camera.load_state_dict(state)
    else:model.load_state_dict(state)
    model.eval().to(device)
    parameters=list(camera.named_parameters())
    active_parameters=None
    rows=json.loads(manifest.read_bytes())
    rows.sort(key=lambda r:(r['title'],r['difficulty'],r['seed']))
    cells={};seeds={};norms={};groups={}
    for (title,rank),cohort in itertools.groupby(rows,key=lambda r:(r['title'],r['difficulty'])):
        label=f'TH{title} '+('Hard' if rank==2 else 'Lunatic')
        pairs=[(r,credit(r,manifest,model,plan['behavior_sha256'],device)[0][:4]) for r in cohort]
        adv=torch.cat([arrays[3] for _,arrays in pairs])
        mean,std=adv.mean(),adv.std(unbiased=False)+1e-7
        vectors=[];counts=[]
        for _,(images,actions,old,advantages) in pairs:
            model.zero_grad(set_to_none=True)
            for start in range(0,len(images),48):
                idx=slice(start,start+48)
                logits=model(images[idx].to(device).float()/255)[:4]
                lp=chosen(logits,actions[idx].to(device)).sum(1)
                weight=((advantages[idx]-mean)/std).to(device)
                loss=-((lp-old[idx].to(device)).exp()*weight).sum()/len(images)
                loss.backward()
            active=[(name,x) for name,x in parameters if x.grad is not None]
            selected=[name for name,_ in active]
            if active_parameters is None:active_parameters=selected
            if selected!=active_parameters:raise ValueError('Policy gradient support changed between folds')
            vec=torch.cat([x.grad.reshape(-1).cpu() for _,x in active])
            if not torch.isfinite(vec).all() or vec.norm()==0:raise ValueError('Degenerate policy gradient')
            vectors.append(vec);counts.append(len(images))
        def average(indices):
            return sum(vectors[i]*counts[i] for i in indices)/sum(counts[i] for i in indices)
        indices=list(range(len(pairs)))
        folds=[indices[::2],indices[1::2]]
        parts=[average(ids) for ids in folds]
        sizes=[sum(counts[i] for i in ids) for ids in folds]
        seeds[label]=[[pairs[i][0]['seed'] for i in ids] for ids in folds]
        cells[label]=(parts[0]*sizes[0]+parts[1]*sizes[1])/sum(sizes)
        splits=[]
        for selected in itertools.combinations(indices,len(indices)//2):
            if 0 not in selected:continue
            other=[i for i in indices if i not in selected]
            splits.append(cosine(average(selected),average(other)))
        norms[label]=dict(decisions=sum(sizes),gradient_l2=float(cells[label].norm()),
                          half_sample_sizes=sizes,within_stratum_split_half_cosine=cosine(*parts),
                          exhaustive_balanced_splits=len(splits),balanced_split_cosines=splits,
                          balanced_split_median_cosine=float(np.median(splits)))
        groups[label]=parts
    names=list(cells)
    g4=torch.stack([v for k,v in cells.items() if not k.startswith('TH11 ')]).mean(0)
    g5=torch.stack(list(cells.values())).mean(0)
    pairwise=[[cosine(cells[i],cells[j]) for j in names] for i in names]
    cross=[[.5*(cosine(groups[i][0],groups[j][1])+cosine(groups[i][1],groups[j][0])) for j in names] for i in names]
    for k,value in cells.items():
        norms[k].update(cosine_with_four_game_mean=cosine(value,g4),cosine_with_five_game_mean=cosine(value,g5),
                        four_game_dot=float(torch.dot(value,g4)),five_game_dot=float(torch.dot(value,g5)))
    out=root/'gradient-audit';out.mkdir(exist_ok=True)
    data=dict(registration=report['protocol_sha256'],behavior=plan['behavior_sha256'],manifest_sha256=digest(manifest),
              interpretation='Post-hoc training diagnosis at the exact deployed behavior checkpoint. Mean policy-loss gradients only, active camera and action-head parameters. Same four-head joint log probability, per-stratum GAE normalization and decision weighting as PPO before clipping becomes active. Excludes value and entropy losses, Adam moments and later epochs. Dots and cosines do not measure game performance. Cross-half estimates divide whole credits within stratum by alternating sorted controller seeds; full-stratum advantage normalization is shared across halves. The separate exhaustive within-stratum result enumerates every balanced partition of whole credits.',
              strata=names,active_parameter_names=active_parameters,fold_seeds=seeds,measurements=norms,pairwise_cosines=pairwise,cross_half_cosines=cross,
              cosine_of_four_and_five_title_mean=cosine(g4,g5))
    (out/'train-gradients.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf8')
    fig,axes=plt.subplots(1,2,figsize=(11.5,5.8),layout='constrained')
    for ax,matrix,heading in zip(axes,(pairwise,cross),('All original training credits','Agreement across disjoint credit halves')):
        im=ax.imshow(matrix,cmap='RdBu',vmin=-1,vmax=1)
        ax.set_xticks(range(len(names)),names,rotation=60,ha='right',fontsize=8)
        ax.set_yticks(range(len(names)),names,fontsize=8)
        ax.set_title(heading,fontsize=11)
    fig.colorbar(im,ax=axes,label='Cosine between policy gradients',shrink=.65)
    fig.suptitle('Initial PPO policy gradients across five original games')
    fig.savefig(out/'train-gradient-agreement.png',dpi=165)
    plt.close(fig)
    print('Cosine of four- and five-game gradients',data['cosine_of_four_and_five_title_mean'],
          'within-cell reliability',{k:round(v['within_stratum_split_half_cosine'],3) for k,v in norms.items()},flush=True)


if __name__=='__main__':
    main()
