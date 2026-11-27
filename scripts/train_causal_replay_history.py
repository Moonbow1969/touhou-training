"""Compare genuinely chronological visual histories to an equal-parameter repeated-current control."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze
from scripts.train_human_camera_policy import evaluation,load_replay
from scripts.train_replay_camera_encoder import CameraPolicy
from scripts.train_portable_full_sft import WIDTHS


class ResidualHistory(nn.Module):
    def __init__(self):
        super().__init__()
        self.memory=nn.GRU(128,96,batch_first=True)
        self.output=nn.Linear(96,sum(WIDTHS))
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,seq,current):
        _,h=self.memory(seq)
        return current+self.output(h[-1])


def paths(data):
    ranked=json.loads((data/'ranked-multigame-sft/design.json').read_bytes())
    base=json.loads((data/ranked['base_split']).read_bytes())['split']
    result={r['sha256']:data/r['file'] for part in ('train','held') for r in base[part]}
    extra=json.loads((data/ranked['expanded_manifest']).read_bytes())['records']
    result.update({r['archive_sha256']:data/r['path'] for r in extra})
    for role in ('train','held'):
        key=data/f'human-retail17/{role}-qualified-02-stages-frozen.json'
        result.update({r['archive_sha256']:data/r['path'] for r in json.loads(key.read_bytes())['qualified']})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    d=p.parse_args().data.resolve();root=d/'four-title-causal-history';root.mkdir(exist_ok=True)
    parent=d/'specialist-vs-shared/merged-equal'
    split=json.loads((d/'specialist-vs-shared/protocol.json').read_bytes())
    protocol=dict(parent_sha256=digest(parent/'policy.ts'),train=split['train'],held=split['held'],seed=265173,
        history_observations=8,native_ticks_between_observations=4,updates=6000,batch_size=128,lr=.0003,
        design='Freeze the identical merged human camera encoder and action logits. Train the same 96-unit GRU residual and all four residual action heads in both arms. One repeats the current visual embedding eight times; the other uses eight strictly consecutive observation embeddings, spaced by four native ticks. Both see identical final images, labels and sampled rows. No human previous action, future pixel, coordinate, geometry or automatic controls. Sample an equal 16 original rows from each of eight game/rank cells per update. Freeze the final checkpoint after 6000 updates.',
        endpoints='Repeated development humans; compare chronological history to the equal-capacity current-image control and the frozen base. Primary is change-of-direction NLL macro-averaged across the four titles. A candidate passes to independent original evaluation only if chronological history beats repeated-current on change NLL in at least three titles and macro all-decision accuracy is at least as high. Report direction starts, turns, button metrics and exact independent player counts.')
    freeze(root/'protocol.json',protocol)
    torch.set_num_threads(2);torch.manual_seed(protocol['seed'])
    device='cuda' if torch.cuda.is_available() else 'cpu'
    camera=CameraPolicy()
    state=torch.load(parent/'camera-policy.pt',map_location='cpu',weights_only=True)
    if any('policy.spatial_context.' in key for key in state):camera.add_spatial_context()
    camera.load_state_dict(state);camera=camera.to(device).eval()
    locator=paths(d);stages={};valid={};groups={'train':set(),'held':set()};summaries={}
    latents=[];base_logits=[];actions=[];offset=0
    for role in ('train','held'):
        for title,hashes in split[role].items():
            for source in hashes:
                path=locator[source]
                row=load_replay(path)
                if row['sha256']!=source or row['title']!=title:raise ValueError('Unregistered original camera stage')
                groups[role].add(row['group'])
                frame=row['native_frame']
                windows=np.lib.stride_tricks.sliding_window_view(frame,8)
                idx=np.flatnonzero((np.diff(windows,axis=1)==4).all(1))+7
                if not len(idx):raise ValueError('Missing contiguous visual history')
                with torch.no_grad():
                    for batch in torch.from_numpy(row['raw']).split(64):
                        features=camera.policy[:-1](camera.spatial(batch.to(device).float()/255))
                        latents.append(features.detach());base_logits.append(camera.policy[-1](features).detach())
                actions.append(torch.from_numpy(row['action']).to(device))
                group=(title,row['difficulty'])
                if role=='train':valid.setdefault(group,[]).append(idx+offset)
                else:stages[source]=dict(indices=idx+offset,group=row['group'],title=title,action=row['action'][idx],prior=row['prior'][idx])
                summaries[source]=dict(role=role,title=title,player=row['group'],difficulty=row['difficulty'],usable_history_rows=len(idx))
                offset+=len(frame)
    if groups['train']&groups['held'] or len(valid)!=8 or {r for _,r in valid}!={'Hard','Lunatic'}:
        raise ValueError(f'Unqualified replay partitions or training ranks {list(valid)}')
    latent=torch.cat(latents);current=torch.cat(base_logits);labels=torch.cat(actions)
    del latents,base_logits,actions
    banks=[torch.as_tensor(np.concatenate(valid[k]),device=device) for k in sorted(valid)]
    lag=torch.arange(-7,1,device=device)
    def history(ids,seq):
        return latent[ids[:,None]+lag[None,:]] if seq else latent[ids,None].expand(-1,8,-1)
    models={}
    for arm in ('repeated-current','chronological'):
        torch.manual_seed(protocol['seed'])
        model=ResidualHistory().to(device)
        rng=torch.Generator(device=device).manual_seed(protocol['seed'])
        opt=torch.optim.AdamW(model.parameters(),lr=protocol['lr'])
        log=[]
        for step in range(1,protocol['updates']+1):
            ids=torch.cat([bank[torch.randint(len(bank),(16,),generator=rng,device=device)] for bank in banks])
            pred=model(history(ids,arm=='chronological'),current[ids]).split(WIDTHS,1)
            loss=sum(F.cross_entropy(h,labels[ids,k],label_smoothing=.025) for k,h in enumerate(pred))
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            if step==1 or step%500==0:
                log.append(dict(step=step,loss=float(loss.detach().cpu())))
                print(arm,step,log[-1]['loss'],flush=True)
        folder=root/arm;folder.mkdir(exist_ok=True)
        model=model.eval()
        script=torch.jit.trace(model,(history(ids[:1],arm=='chronological'),current[ids[:1]]))
        script.save(str(folder/'residual.ts'))
        torch.testing.assert_close(script(history(ids[:4],arm=='chronological'),current[ids[:4]]),model(history(ids[:4],arm=='chronological'),current[ids[:4]]))
        (folder/'train-log.json').write_text(json.dumps(log,indent=2)+'\n')
        models[arm]=model
    results={}
    with torch.no_grad():
        for arm in ('base','repeated-current','chronological'):
            results[arm]={}
            for source,row in stages.items():
                logits=[]
                for ids in torch.as_tensor(row['indices'],device=device).split(128):
                    logits.append(current[ids] if arm=='base' else models[arm](history(ids,arm=='chronological'),current[ids]))
                probs=[x.softmax(1).cpu().numpy() for x in torch.cat(logits).split(WIDTHS,1)]
                results[arm][source]=evaluation(row,probs)
    pooled={}
    for title in split['held']:
        pooled[title]={}
        for arm,metrics in results.items():
            group=[metrics[s] for s in split['held'][title]]
            def summary(c):
                n=sum(m[c]['n'] for m in group)
                return dict(n=n,accuracy=sum(m[c]['correct'] for m in group)/n,nll=sum(m[c]['n']*m[c]['movement_nll'] for m in group if m[c]['n'])/n)
            pooled[title][arm]={c:summary(c) for c in ('all','change','start','turn')}
    gains=sum(row['chronological']['change']['nll']<row['repeated-current']['change']['nll'] for row in pooled.values())
    delta=np.mean([row['chronological']['all']['accuracy']-row['repeated-current']['all']['accuracy'] for row in pooled.values()])
    final=dict(registered_protocol_sha256=digest(root/'protocol.json'),parent_sha256=protocol['parent_sha256'],
        training_presentations_per_arm=protocol['updates']*protocol['batch_size'],source_stages=summaries,
        unique_train_history_rows=sum(len(bank) for bank in banks),unique_held_history_rows=sum(len(r['indices']) for r in stages.values()),
        unique_players={k:len(v) for k,v in groups.items()},model_sha256={a:digest(root/a/'residual.ts') for a in models},
        pooled_by_title=pooled,per_stage_metrics=results,change_nll_improved_titles=gains,macro_all_accuracy_difference=delta,
        original_evaluation_gate=bool(gains>=3 and delta>=0))
    (root/'report.json').write_text(json.dumps(final,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print('Completed causal history comparison',gains,delta,final['original_evaluation_gate'],flush=True)


if __name__=='__main__':
    main()
