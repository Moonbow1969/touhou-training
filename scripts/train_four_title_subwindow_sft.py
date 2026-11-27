"""Train and evaluate one- and two-command visual policies on synchronized original replay windows."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scripts.checkpoint_storage import digest
from scripts.train_causal_replay_history import paths
from scripts.train_five_original_games import freeze
from scripts.train_human_camera_policy import camera_labels, load_replay
from scripts.train_portable_full_sft import WIDTHS
from scripts.train_replay_camera_encoder import CameraPolicy


class SubwindowHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.output=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,2*sum(WIDTHS)))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self,latent,original):
        return original.unsqueeze(1)+self.output(latent).reshape(-1,2,sum(WIDTHS))


def stage(path,model,device):
    row=load_replay(path)
    with np.load(path,allow_pickle=False) as z:
        frames=z['native_frame'].reshape(-1)
        if np.any(np.diff(frames.astype('int64'))<=0):
            raise ValueError('Original recording clock not strictly increasing')
        idx=np.searchsorted(frames,row['native_frame'])
        if not np.array_equal(frames[idx],row['native_frame']):raise ValueError('Input and camera clock mismatch')
        words=z['input_chunk'][idx]
    focus=4 if row['title']=='TH10' else 8
    if not np.array_equal(row['action'],camera_labels(words,focus)):raise ValueError('Original labels disagree')
    labels2=np.stack([camera_labels(words[:,j:j+2],focus) for j in (0,2)],axis=1)
    native=np.stack([camera_labels(words[:,j:j+1],focus) for j in range(4)],axis=1)
    feats=[];logits=[]
    with torch.inference_mode():
        for x in torch.from_numpy(row['raw']).split(64):
            h=model.policy[:-1](model.spatial(x.to(device).float()/255))
            feats.append(h.cpu());logits.append(model.policy[-1](h).cpu())
    return dict(group=row['group'],title=row['title'],difficulty=row['difficulty'],sha256=row['sha256'],
                features=torch.cat(feats),base=torch.cat(logits),one=torch.from_numpy(row['action']),
                two=torch.from_numpy(labels2),native=torch.from_numpy(native))


def scored(row,heads,device):
    result={}
    truth=row['native'].to(device)
    turn=(truth[:,:,0]!=truth[:,:1,0])
    for name,model in heads.items():
        with torch.no_grad():
            logits=model(row['features'].to(device),row['base'].to(device))
            if name=='four-tick':
                logits=logits.mean(1,keepdim=True).expand(-1,4,-1)
            else:
                logits=logits.repeat_interleave(2,dim=1)
            losses=[];hits=[]
            for i,head in enumerate(logits.split(WIDTHS,dim=2)):
                y=truth[:,:,i]
                losses.append(float(F.cross_entropy(head.flatten(0,1),y.flatten())))
                hits.append(int((head.argmax(2)==y).sum()))
            pred=logits[:,:,:9].argmax(2)
            result[name]=dict(ticks=truth.shape[0]*4,per_head_mean_nll=losses,per_head_correct=hits,
                              moved_within_window_ticks=int(turn.sum()),moved_within_window_correct=int(((pred==truth[:,:,0])&turn).sum()))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--fresh-held',action='store_true')
    a=p.parse_args();d=a.data.resolve();root=d/'four-title-subwindow-sft';root.mkdir(exist_ok=True)
    base=d/'ranked-multigame-sft/four-title-expanded-twelve'
    earlier=json.loads((d/'specialist-vs-shared/protocol.json').read_bytes())
    source=json.loads((base/'report.json').read_bytes())
    locator=paths(d)
    manifest=d/'human-retail17-wave2/train-original-synchronized-frozen.json'
    for row in json.loads(manifest.read_bytes())['qualified']:locator[row['archive_sha256']]=d/row['path']
    plan=dict(parent=digest(base/'camera-policy.pt'),training_report=digest(base/'report.json'),
        training=[r['sha256'] for r in source['train']],development=earlier['held'],updates=3000,per_cell=8,
        seed=271699,learning_rate=.0003,
        design='Freeze the same four-title image encoder and old policy logits. Fit identical 128-unit residual heads with two outputs for all four controls. One arm supervises both outputs on the existing four-tick label and averages their logits for a single command. The other supervises each two-tick subwindow separately and issues the two commands in sequence. At inference both see only the same initial image and previous-image difference. Optimize each head equally on the same 192,000 samples drawn equally from eight game/rank cells. No image from the second subwindow or human previous action is input.',
        primary='Compare mean movement negative log likelihood per actual native tick, macro over development players within title and equally over titles. A follow-up native test requires improvement on at least three of four development titles and on the independent new TH17 cohort. Also report all-head NLL, direction accuracy, and correctly predicting human input after a direction change inside a window. Fresh four TH17 authors are assessed after independent synchronization and after checkpoints freeze. Comparing predictors at fixed human screens measures imitation.')
    freeze(root/'protocol.json',plan)
    torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=CameraPolicy()
    state=torch.load(base/'camera-policy.pt',weights_only=True,map_location='cpu')
    if any('policy.spatial_context.' in k for k in state):model.add_spatial_context()
    model.load_state_dict(state);model.to(device).eval()
    arms=('four-tick','two-subwindows')
    fitted={}
    for arm in arms:
        torch.manual_seed(plan['seed'])
        fitted[arm]=SubwindowHead().to(device)
    if a.fresh_held:
        registered=json.loads((root/'report.json').read_bytes())
        for name,head in fitted.items():
            weights=root/f'{name}.pt'
            if digest(weights)!=registered['weights'][name]:raise ValueError('Fresh held checkpoint differs')
            head.load_state_dict(torch.load(weights,weights_only=True,map_location=device))
        receipt=d/'human-retail17-wave2-held/held-original-synchronized-frozen.json'
        held_source=json.loads(receipt.read_bytes())
        if any(r['split']!='held' or r['player_group'].casefold() in set(registered['training_players']) for r in held_source['qualified']):
            raise ValueError('Unexpected split in fresh human evaluations')
        held_files=[d/r['path'] for r in held_source['qualified']]
        if any(digest(f)!=r['archive_sha256'] for f,r in zip(held_files,held_source['qualified'])):raise ValueError('Changed held camera')
    else:
        banks=defaultdict(list);players=set()
        for row in source['train']:
            one=stage(locator[row['sha256']],model,device)
            if one['sha256']!=row['sha256'] or len(one['one'])!=row['decisions']:raise ValueError('SFT windows differ')
            players.add(one['group'])
            banks[(one['title'],one['difficulty'])].append({k:one[k] for k in ('features','base','one','two')})
        data={k:{field:torch.cat([v[field] for v in rs]).to(device) for field in rs[0]} for k,rs in sorted(banks.items())}
        if len(data)!=8 or sum(len(v['one']) for v in data.values())!=148415:raise ValueError('Incomplete training set')
        optim={k:torch.optim.AdamW(head.parameters(),lr=plan['learning_rate']) for k,head in fitted.items()}
        rng=torch.Generator().manual_seed(plan['seed']);history=[]
        for step in range(1,plan['updates']+1):
            mini=defaultdict(list)
            for cell in data.values():
                ix=torch.randint(len(cell['one']),(8,),generator=rng).to(device)
                for k,v in cell.items():mini[k].append(v[ix])
            batch={k:torch.cat(items) for k,items in mini.items()}
            metrics={}
            for name,head in fitted.items():
                logits=head(batch['features'],batch['base'])
                y=batch['one'][:,None,:].expand(-1,2,-1) if name=='four-tick' else batch['two']
                loss=sum(F.cross_entropy(h.flatten(0,1),y[:,:,j].reshape(-1)) for j,h in enumerate(logits.split(WIDTHS,dim=2)))
                optim[name].zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(),1.);optim[name].step()
                metrics[name]=float(loss.detach())
            if step==1 or step%250==0:
                history.append(dict(step=step,loss=metrics))
                print('SFT',step,metrics,flush=True)
        for name,head in fitted.items():torch.save(head.state_dict(),root/f'{name}.pt')
        held_files=[locator[k] for hashes in earlier['held'].values() for k in hashes]
    scores=[]
    for file in held_files:
        row=stage(file,model,device)
        if not a.fresh_held and row['group'] in players:raise ValueError('Held development player overlaps training')
        scores.append(dict(player=row['group'],sha256=row['sha256'],title=row['title'],difficulty=row['difficulty'],metrics=scored(row,fitted,device)))
    gaps={}
    for title in sorted({r['title'] for r in scores}):
        player_values=[]
        for player in sorted({r['player'] for r in scores if r['title']==title}):
            items=[r['metrics'] for r in scores if r['title']==title and r['player']==player]
            n=sum(x[arms[0]]['ticks'] for x in items)
            player_values.append(sum((x[arms[1]]['per_head_mean_nll'][0]-x[arms[0]]['per_head_mean_nll'][0])*x[arms[0]]['ticks'] for x in items)/n)
        gaps[title]=float(np.mean(player_values))
    output=dict(registered=digest(root/'protocol.json'),weights={name:digest(root/f'{name}.pt') for name in arms},by_stage=scores,
                movement_nll_difference_by_title=gaps,improved_titles=sum(v<0 for v in gaps.values()))
    if a.fresh_held:output['held_acquisition']=digest(receipt)
    else:output.update(history=history,training_players=sorted(players),distinct_decisions=148415,sampled_windows_per_arm=192000)
    freeze(root/('fresh-four.json' if a.fresh_held else 'report.json'),output)
    print('Held movement NLL changes',gaps,flush=True)


if __name__=='__main__':
    main()
