"""Frozen state-only action network. No reference data is accepted by the actor."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class NativeClock:
    def __init__(self):
        self.previous=None
        self.change=0

    def __call__(self,stage,frame,boss_max,spell):
        identity=(stage,boss_max,spell)
        if self.previous!=identity:
            self.change=frame
            self.previous=identity
        return max(0,frame-self.change)


def features(context,bullets,stage,frame,max_hp,spell,vulnerable,elapsed,state,previous_bomb,stock):
    current=np.asarray(context[:10],np.float32).copy()
    current[8]=np.clip(current[8],0,1)
    nearest=bullets[:3,:7].reshape(-1)
    physical=np.r_[current,nearest,np.eye(6,dtype=np.float32)[stage-1],max_hp/20000,
                   np.log1p(max(0,max_hp))/10,(spell+1)/100,spell>=0,vulnerable]
    return (np.asarray(physical,np.float32),np.asarray([frame,elapsed],np.float32),
            np.asarray([state,previous_bomb,stock],np.float32))


class StateActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("period",torch.logspace(np.log10(16),np.log10(24576),40))
        self.timing=nn.Sequential(nn.Linear(42+162,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,128),nn.SiLU())
        self.goal=nn.Linear(128,2)
        self.controls=nn.Sequential(nn.Linear(128+42+2,192),nn.SiLU(),nn.Linear(192,96),nn.SiLU(),nn.Linear(96,13))
        self.bomb=nn.Sequential(nn.Linear(11,32),nn.SiLU(),nn.Linear(32,2))

    def forward(self,x,clock,raw):
        phase=clock.unsqueeze(-1)*(2*torch.pi/self.period)
        line=torch.cat((clock/12000,phase.sin().flatten(1),phase.cos().flatten(1)),1)
        # Temporal features learn a continuous representation of the current encounter.
        # The waypoint regression is auxiliary. Action decoding is learned end to end.
        context=x.clone()
        context[:,:31]=0
        h=self.timing(torch.cat((context,line),1))
        goal=self.goal(h)
        logits=self.controls(torch.cat((x,h,goal),1))
        raw_state=torch.nn.functional.one_hot(raw[:,0].long().clamp(0,8),9).to(x.dtype)
        bomb=self.bomb(torch.cat((raw_state,raw[:,1:2],raw[:,2:3]/8),1))
        return (*logits.split((9,2,2),1),bomb,goal)


class StockActor:
    stage_clock=False
    motion_spans=None
    retain_recent_motion=False

    def __init__(self,path):
        weights=torch.load(path,map_location="cpu",weights_only=True)
        self.net=StateActor().eval()
        self.net.load_state_dict(weights["state_dict"])
        self.environment=None
        self.credit=None

    @torch.inference_mode()
    def predict(self,observation,deterministic=False):
        env=self.environment; s=env.snapshot
        if self.credit!=env.reset_count:
            self.clock=NativeClock(); self.credit=env.reset_count
        b=s.boss; hp=b.max_hp if b else 0; spell=s.spell_id if s.spell_id is not None else -1
        age=self.clock(s.stage,env.last_frame,hp,spell)
        inputs=features(observation["context"],observation["bullets"],s.stage,env.last_frame,hp,spell,
                        bool(b and b.vulnerable),age,s.player_state,int("x" in env.held),env.game.read_game_val("bombs"))
        logits=self.net(*(torch.from_numpy(v[None]) for v in inputs))
        return np.asarray([v.argmax(1).item() for v in logits[:4]],np.int64),None


def main():
    from scripts import evaluate_credit_story as story
    from scripts.checkpoint_storage import digest
    from scripts.evaluate_sft_native import Arm,output_lock
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights",required=True,type=Path);p.add_argument("--out",required=True,type=Path)
    p.add_argument("--seeds",required=True,type=int,nargs="+");p.add_argument("--target",choices=story.TARGETS,default="stage2")
    p.add_argument("--parakit",type=Path,default=Path(__file__).resolve().parents[2]/"parakit")
    p.add_argument("--start-frame",type=int,default=30);p.add_argument("--focus-wait",type=int,default=90)
    p.add_argument("--game-over-default-reference",type=Path,required=True);p.add_argument("--game-over-retry-reference",type=Path,required=True)
    p.add_argument("--menu-crop",type=int,action="append",nargs=4,required=True)
    p.add_argument("--menu-max-error",type=float,default=.035);p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    args.limit=story.LIMITS[args.target];args.crossover=[];args.timing_policy=[];args.restart_hook=None
    args.transition_budget_mib=128;args.transition_reserve_mib=2048
    torch.set_num_threads(1)
    arms={"state_student":Arm(StockActor(args.weights),False,dict(kind="state-only neural actor",sha256=digest(args.weights)))}
    spec=story.run_specification(args,arms);spec["controller"]["kind"]="greedy four-head state-only student"
    spec["runtime"]["source_sha256"]["scripts.state_route_student"]=digest(Path(__file__))
    from scripts.native_story_transitions import configure
    archive=configure(args,spec)
    if not args.resume:args.out.mkdir(parents=True,exist_ok=False)
    menu=story.GameOverMenu(args.game_over_default_reference,args.game_over_retry_reference,args.menu_crop,args.menu_max_error)
    with output_lock(args.out):
        report=story.evaluate(args,arms,spec,args.out/"report.json",menu,None,transition_archive=archive)
    print(json.dumps(dict(status=report["status"])))


if __name__=="__main__":main()
