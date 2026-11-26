"""State-only neural action policy trained on privileged hybrid teacher labels."""
from pathlib import Path
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO.parents[1]
OUT=Path(os.environ.get('TH14_HYBRID_STUDENT_ARTIFACT_DIR',ROOT/'outputs/th14-hybrid-dodge-student-261125')).resolve()
from scripts.relative_phase_student import StockRelativeActor


class AnticipatoryContact(nn.Module):
    """Inference-only instance of the frozen learned 96-entity contact encoder."""
    def __init__(self):
        super().__init__()
        self.point=nn.Sequential(nn.Linear(12,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,19))

    def forward(self,bullets,context):
        tokens=bullets[:,:,:6]*bullets.new_tensor((6.,7.,2.,2.,4.,1.))
        values=torch.cat((tokens,context[:,None].expand(-1,bullets.shape[1],-1)),dim=-1)
        active=(bullets[:,:,6]>0)&(bullets[:,:,5]>0)&(bullets[:,:,4]>0)
        return self.point(values).masked_fill(~active[:,:,None],-14.).amax(1)


class MovementStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.state=nn.Sequential(nn.Linear(18+36+10+6,128),nn.SiLU(),nn.Linear(128,96),nn.SiLU())
        self.keep=nn.Linear(96,18)
        self.replacement=nn.Linear(96,36)
        self.margin=nn.Linear(96,18)
        self.register_buffer('keep_offset',torch.zeros(()))

    def encode(self,x):
        h=self.state(x)
        return self.keep(h),self.replacement(h).reshape(-1,2,18),self.margin(h)

    def logits(self,x,prior):
        keep,alternate,_=self.encode(x)
        rows=torch.arange(len(prior),device=prior.device)
        lk=keep[rows,prior]+self.keep_offset
        position=torch.arange(18,device=prior.device)[None]==prior[:,None]
        alt=alternate[rows,prior//9].masked_fill(position,-1e6).log_softmax(1)
        return torch.where(position,F.logsigmoid(lk)[:,None],F.logsigmoid(-lk)[:,None]+alt)


def feature_vectors(contact,bullets,ctx,stage):
    shot=ctx.new_tensor((1.,0.,0.)).expand(len(ctx),-1)
    risk=contact(bullets,torch.cat((ctx[:,:3],shot),1))[:,:18].clamp(-15,12)/6
    # The inherited point encoder pools all 96 raw entities. The closest six also reach the new head directly.
    nearest=(bullets[:,:6,:6]*bullets.new_tensor([5,6,2,2,3,1])).clamp(-4,4).flatten(1)
    return torch.cat([risk,nearest,ctx[:,:10],F.one_hot(stage.long()-1,6).float()],1)


class StockHybridStudent(StockRelativeActor):
    def __init__(self,path):
        super().__init__(path)
        weights=torch.load(path,weights_only=True,map_location='cpu')
        if weights['schema']!='th14-hybrid-privileged-state-only-v1':
            raise ValueError('Incompatible neural policy.')
        self.contact=AnticipatoryContact().eval()
        self.contact.load_state_dict(weights['contact'])
        self.movement=MovementStudent().eval()
        self.movement.load_state_dict(weights['movement'])

    @torch.inference_mode()
    def predict(self,observation,deterministic=False):
        action,_=super().predict(observation,deterministic)
        x=feature_vectors(self.contact,torch.from_numpy(observation['bullets'][None]),
                          torch.from_numpy(observation['context'][None]),torch.tensor([self.environment.snapshot.stage]))
        prior=torch.tensor([int(action[0])+9*int(action[1])])
        joint=int(self.movement.logits(x,prior).argmax())
        action[:2]=[joint%9,joint//9]
        return action,None
