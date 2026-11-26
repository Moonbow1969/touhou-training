"""State-only actor with observable event clocks and a learned internal waypoint."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class EventClock:
    def __init__(self):
        self.stage=None

    def update(self,stage,frame,maxhp,spell,visible):
        if self.stage!=stage:
            self.stage=stage;self.last=None;self.prior=(0,-1);self.encounters=0
            self.seen=-1000000;self.encounter_start=frame;self.change=frame
        event=(maxhp,spell)
        if self.last!=event:
            if self.last is not None:self.prior=self.last
            self.last=event;self.change=frame
        if visible:
            if frame-self.seen>64:
                self.encounter_start=frame;self.encounters+=1
            self.seen=frame
        return np.asarray([frame-self.change,frame-self.encounter_start,self.encounters,self.prior[0],self.prior[1]],np.float32)


class RelativeActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("periods",torch.logspace(np.log10(12),np.log10(32768),64))
        self.register_buffer("hp_periods",torch.tensor([127.,509.,2039.,8191.]))
        self.time_net=nn.Sequential(nn.Linear(2*129+6+2+16+6,256),nn.SiLU(),nn.Linear(256,256),
            nn.SiLU(),nn.Linear(256,128),nn.SiLU())
        self.waypoint=nn.Linear(128,2)
        self.preference=nn.Linear(128,4)
        self.relative=nn.Linear(4,2)
        with torch.no_grad():
            self.relative.weight.copy_(torch.tensor([[12.,0.,-12.,0.],[0.,28.,0.,-28.]]));self.relative.bias.zero_()
        self.motor=nn.Sequential(nn.Linear(2+2+4+2,80),nn.SiLU(),nn.Linear(80,80),nn.SiLU(),nn.Linear(80,18))
        self.shot=nn.Sequential(nn.Linear(128+4+6,48),nn.SiLU(),nn.Linear(48,2))
        self.bomb=nn.Sequential(nn.Linear(11,32),nn.SiLU(),nn.Linear(32,2))

    def encode(self,x,event,hp):
        ages=event[:,:2].clamp(0,30000)
        f=ages.unsqueeze(-1)*(2*torch.pi/self.periods)
        time=torch.cat((ages/12000,f.sin().flatten(1),f.cos().flatten(1)),1)
        size=torch.stack((x[:,37]*20000,event[:,3]),1)
        hp_codes=size.unsqueeze(-1)*(2*torch.pi/self.hp_periods)
        extra=torch.stack((event[:,2]/5,event[:,4]/100,x[:,39],x[:,40],x[:,41],hp),1)
        h=self.time_net(torch.cat((time,x[:,31:37],size/20000,hp_codes.sin().flatten(1),hp_codes.cos().flatten(1),extra),1))
        return h,self.waypoint(h),self.preference(h)

    def decode(self,x,h,target,preference):
        prob=torch.cat((preference[:,:2].softmax(1),preference[:,2:].softmax(1)),1)
        pos=x[:,:2]
        offset=self.relative(torch.cat((target,pos),1)).clamp(-6,6)
        joint=self.motor(torch.cat((offset,pos,prob,x[:,2:4]),1))
        shoot=self.shot(torch.cat((h,prob,x[:,36:42]),1))
        return joint,shoot

    def bomb_logits(self,raw):
        state=F.one_hot(raw[:,0].long().clamp(0,8),9).to(raw.dtype)
        return self.bomb(torch.cat((state,raw[:,1:2],raw[:,2:3]/8),1))

    def forward(self,x,event,hp,raw):
        h,target,pref=self.encode(x,event,hp)
        joint,shoot=self.decode(x,h,target,pref)
        return joint,shoot,self.bomb_logits(raw),target,pref


class StockRelativeActor:
    stage_clock=False
    motion_spans=None
    retain_recent_motion=False

    def __init__(self,path):
        from scripts.state_route_student import features
        self.features=features
        self.model=RelativeActor().eval()
        self.model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True)["state_dict"])
        self.environment=None;self.credit=None

    @torch.inference_mode()
    def predict(self,observation,deterministic=False):
        env=self.environment;s=env.snapshot;boss=s.boss;frame=env.last_frame
        if self.credit!=env.reset_count:
            self.credit=env.reset_count;self.clock=EventClock()
        size=boss.max_hp if boss else 0;sp=s.spell_id if s.spell_id is not None else -1
        e=self.clock.update(s.stage,frame,size,sp,observation["context"][8]>0)
        x,_,raw=self.features(observation["context"],observation["bullets"],s.stage,frame,size,sp,
            bool(boss and boss.vulnerable),e[0],s.player_state,int("x" in env.held),env.game.read_game_val("bombs"))
        hp=np.asarray([np.clip(boss.hp/max(1,size),0,1) if boss else 0],np.float32)
        j,z,b,_,_=self.model(torch.from_numpy(x[None]),torch.from_numpy(e[None]),torch.from_numpy(hp),torch.from_numpy(raw[None]))
        joint=j.argmax(1).item()
        return np.array([joint%9,joint//9,z.argmax(1).item(),b.argmax(1).item()],np.int64),None
