"""Full-control replay SFT and corresponding portable actor-critic checkpoint format."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from scripts import pretrain_policy as bc
from scripts.checkpoint_storage import digest

WIDTHS=(9,2,2,2)


class FullActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.first=nn.Linear(110,96)
        self.second=nn.Linear(96,96)
        self.output=nn.Linear(96,16)

    def encode(self,x):
        return torch.tanh(self.second(torch.tanh(self.first(x))))

    def forward(self,x):
        y=self.output(self.encode(x))
        return y[:,:15].split(WIDTHS,dim=-1),y[:,15]

    def export(self,path,magic=b'THFULL01'):
        with path.open('xb') as f:
            f.write(magic)
            for p in self.parameters():
                f.write(p.detach().cpu().numpy().astype('<f4').tobytes())

    @classmethod
    def from_bin(cls,path):
        model=cls()
        with path.open('rb') as f, torch.no_grad():
            if f.read(8) not in (b'THFULL01',b'THVIS001'):raise ValueError('Unrecognized checkpoint.')
            for p in model.parameters():
                a=np.frombuffer(f.read(p.numel()*4),dtype='<f4').copy().reshape(tuple(p.shape))
                p.copy_(torch.from_numpy(a))
            if f.read():raise ValueError('Trailing checkpoint data')
        return model


def trajectory(path,row,teacher,device,visual_position=False):
    with np.load(path,allow_pickle=False) as z:
        frame=z['stage_frame']
        start=np.arange(0,len(frame)-1,2)
        start=start[frame[start+1]-frame[start]==4]
        chunks=z['expert_action_chunk']
        chunk=np.concatenate([chunks[start],chunks[start+1]],axis=1)
        y=bc.label_actions(chunk)
        soft=bc.movement_distribution(chunk)
        h=np.zeros((len(start),110),np.float32)
        for lag in (1,2):
            valid=np.flatnonzero((frame[start][lag:]-frame[start][:-lag])==8*lag)+lag
            prev=y[valid-lag];offset=84+(lag-1)*12
            h[valid,offset+prev[:,0]]=1
            h[valid,offset+9:offset+12]=prev[:,1:]
        age=frame[start]/8
        h[:,108]=np.minimum(age/120,1);h[:,109]=np.minimum(age/1800,1)
        if visual_position:
            position=z["context"][start,:2]
            h[:,:2]=position
            previous=np.flatnonzero(np.diff(frame[start])==8)+1
            delta=(position[previous]-position[previous-1])*np.array((192.,448.))/32
            h[previous,2:4]=delta.clip(-1,1)
    # The teacher uses validated structured replay observations; training never exposes
    # its native state to the portable actor deployed in the original game.
    b,c,_=bc.read_shard((path,row),True,motion_spans=(4,12,36))
    pred=[]
    with torch.inference_mode():
        for k in range(0,len(start),384):
            idx=start[k:k+384]
            heads=bc.predict_heads(teacher,torch.from_numpy(b[idx]).to(device),torch.from_numpy(c[idx]).to(device))
            pred.append(torch.cat([logits.softmax(1) for logits in heads],dim=1).cpu())
    return (torch.from_numpy(h),torch.from_numpy(y),torch.from_numpy(soft),torch.cat(pred))


def collect(rows,args,teacher,device):
    datasets={}
    for r in rows:
        path=args.data/r['file']
        if digest(path)!=r['sha256']:raise ValueError(f'Shard fingerprint changed: {path.name}')
        datasets.setdefault(r['title'],[]).append(trajectory(path,r,teacher,device,args.visual_position))
    return {t:tuple(torch.cat(part).to(device) for part in zip(*shards)) for t,shards in datasets.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','data','teacher','out'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--steps',type=int,default=3000);p.add_argument('--seed',type=int,default=71231)
    p.add_argument('--visual-position',action='store_true',
                   help='Train on native human position and displacement, supplied by a learned visual locator at play.')
    args=p.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed);torch.set_num_threads(2)
    manifest=json.loads(args.manifest.read_bytes())
    fit,hold=manifest['train_manifest'],manifest['evaluation_manifest']
    if {r['player_group'] for r in fit}&{r['player_group'] for r in hold}:raise ValueError('Player overlap')
    arms=[a for a in manifest['arms'] if a.get('sha256')==digest(args.teacher)]
    if len(arms)!=1 or set(arms[0]['train_titles'])!={13,14,17}:raise ValueError('Unverified full-action human teacher')
    teacher=MultiInputActorCriticPolicy(bc.observation_space(True),bc.action_space(),lambda _:3e-4,**bc.network_kwargs(256)).to(device)
    teacher.load_state_dict(torch.load(args.teacher,weights_only=True,map_location=device));teacher.eval()
    train=collect(fit,args,teacher,device);test=collect(hold,args,teacher,device)
    actor=FullActor().to(device)
    with torch.no_grad():
        actor.first.weight[:,:84].zero_()
        actor.output.weight[15].zero_();actor.output.bias[15].zero_()
    opt=torch.optim.AdamW(actor.parameters(),lr=.0006)
    history=[]
    for idx in range(1,args.steps+1):
        parts=[]
        for t in sorted(train):
            arrays=train[t]
            ii=torch.randint(len(arrays[0]),(128,),device=device)
            parts.append([item[ii] for item in arrays])
        x,y,soft,teacher_p=map(torch.cat,zip(*parts))
        heads,_=actor(x)
        targets=[soft]+[F.one_hot(y[:,k],2).float() for k in (1,2,3)]
        loss=sum(-((.85*a+.15*b)*logit.log_softmax(1)).sum(1).mean() for a,b,logit in zip(targets,teacher_p.split(WIDTHS,1),heads))
        opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),1);opt.step()
        if idx==1 or idx%250==0:history.append(dict(step=idx,loss=loss.item()))
    held={}
    with torch.no_grad():
        for t,(x,y,_,_) in test.items():
            heads,_=actor(x)
            held[t]=[dict(accuracy=(h.argmax(1)==y[:,k]).float().mean().item(),
                          nll=F.cross_entropy(h,y[:,k]).item(),
                          target_counts=torch.bincount(y[:,k],minlength=WIDTHS[k]).tolist(),
                          predicted_probability=h.softmax(1).mean(0).tolist()) for k,h in enumerate(heads)]
    args.out.mkdir(parents=True,exist_ok=False)
    actor.export(args.out/'policy.bin',b'THVIS001' if args.visual_position else b'THFULL01')
    (args.out/'report.json').write_text(json.dumps(dict(teacher=digest(args.teacher),manifest=digest(args.manifest),policy_sha256=digest(args.out/'policy.bin'),seed=args.seed,steps=args.steps,
        raw_train_decisions=sum(r['samples'] for r in fit),paired_eight_tick_train={t:len(v[0]) for t,v in train.items()},
        held_by_title={t:len(v[0]) for t,v in test.items()},held_head_metrics=held,history=history,teacher_weight=.15,
        observation=('Human player XY and last displacement for SFT, image-only learned position and displacement for native evaluation. ' if args.visual_position else
                     'Human structured replays have no original pixel archive; visual columns start at zero weight and are optimized from actual pixels by online PPO. ')+
                    'Two prior executed full controls and elapsed decision estimates.',
        action='Categorical 9-way movement and independent focus, shoot and Bomb categorical heads. All 72 joint controls expressible.'),indent=2)+'\n')

if __name__=='__main__':main()
