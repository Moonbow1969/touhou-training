"""Run a replay-trained movement/focus policy and short PPO updates in original TH10."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from scripts.checkpoint_storage import digest
from scripts.pretrain_transfer_move_focus import POLICY_KWARGS,shared_live
from scripts.train_full_policy import publish_json
from scripts.train_th10_trial_live_ppo import NativeTH10


class SharedNative(NativeTH10):
    motion_sign=False
    def observe(self,state,previous,focus):
        obs=shared_live(super().observe(state,previous,focus),self.motion_sign)
        if state["lives"]!=previous["lives"] or state["stage"]!=previous["stage"]:
            obs["context"][7:9]=0
        return obs


def actor(env, seed, sft):
    model=PPO("MultiInputPolicy",env,seed=seed,device="cpu",n_steps=128,batch_size=64,n_epochs=4,
              gamma=.996,gae_lambda=.95,learning_rate=.00025,ent_coef=.012,target_kl=.03,policy_kwargs=POLICY_KWARGS)
    if sft:
        calibration=json.loads(sft.with_name("report.json").read_bytes())
        if calibration["weights_sha256"]!=digest(sft) or calibration["train_titles"]!=[13,14,17]:
            raise ValueError("Unexpected SFT source or title population.")
        model.policy.load_state_dict(torch.load(sft,weights_only=True,map_location="cpu"))
    return model


def run(args):
    torch.set_num_threads(1)
    args.out.mkdir(exist_ok=False)
    worker=json.loads(args.workers.read_bytes())[args.lane]
    motion_sign=json.loads(args.sft.with_name("report.json").read_bytes()).get("observed_motion_sign",False)
    log=dict(mode=args.mode,host="Momentum",lane=args.lane,status="running",horizon=args.horizon,
             game_sha256=worker["executable_sha256"],source_sha256=digest(Path(__file__)),
             worker_sources=worker["sources"],sft_sha256=digest(args.sft),seed=args.seed,
             reward="1/horizon for each non-injury decision, -1 on first native life decrement",
             controls="Sampled movement and focus, continuous Shoot, no Bomb. Shared observed fields only.",
             observed_motion_sign=motion_sign,failed_attempts=[],updates=[],credits=[])
    journal=args.out/"report.json"
    publish_json(journal,log)
    if args.mode=="train":
        last=None
        for update in range(1,args.updates+1):
            for retry in range(1,4):
                trace=args.out/f"update-{update}-try-{retry}.jsonl.gz"
                with gzip.open(trace,"wt",encoding="utf-8") as stream:
                    env=SharedNative(worker,args.horizon,stream)
                    env.motion_sign=motion_sign
                    try:
                        model=actor(env,args.seed,args.sft) if last is None else PPO.load(last,env=env,device="cpu")
                        model.set_random_seed(args.seed+update)
                        model.learn(128,reset_num_timesteps=last is None)
                    except (RuntimeError,TimeoutError,BrokenPipeError,EOFError,OSError) as exc:
                        log["failed_attempts"].append(dict(update=update,retry=retry,reason=str(exc),trace=trace.name))
                        publish_json(journal,log)
                        if retry==3:
                            raise
                        continue
                    finally:
                        env.close()
                last=args.out/f"update-{update}.zip"
                model.save(last)
                log["updates"].append(dict(update=update,accepted_total=model.num_timesteps,checkpoint=last.name,
                                      sha256=digest(last),trace=trace.name,trace_sha256=digest(trace)))
                publish_json(journal,log)
                break
    else:
        for block,seed in enumerate(args.eval_seeds):
            arms=["sft","ppo" if args.ppo else "random"]
            if block%2:
                arms.reverse()
            for arm in arms:
                for retry in range(1,4):
                    trace=args.out/f"{seed}-{arm}-try{retry}.jsonl.gz"
                    with gzip.open(trace,"wt",encoding="utf-8") as stream:
                        env=SharedNative(worker,args.horizon,stream)
                        env.motion_sign=motion_sign
                        try:
                            if arm=="ppo":
                                parent=json.loads(args.ppo.with_name("report.json").read_bytes())
                                if parent["status"]!="complete" or digest(args.ppo)!=parent["updates"][-1]["sha256"]:
                                    raise ValueError("Missing completed PPO source.")
                                model=PPO.load(args.ppo,env=env,device="cpu")
                            else:
                                model=actor(env,args.seed,args.sft if arm=="sft" else None)
                            obs,_=env.reset(seed=seed)
                            torch.manual_seed(seed)
                            while True:
                                a,_=model.predict(obs,deterministic=False)
                                obs,_,term,trunc,info=env.step(a)
                                if term or trunc:
                                    break
                        except (RuntimeError,TimeoutError,BrokenPipeError,EOFError,OSError) as exc:
                            log["failed_attempts"].append(dict(arm=arm,seed=seed,retry=retry,reason=str(exc),trace=trace.name))
                            publish_json(journal,log)
                            if retry==3:
                                raise
                            continue
                        finally:
                            env.close()
                    log["credits"].append(dict(arm=arm,seed=seed,trace=trace.name,sha256=digest(trace),**info))
                    publish_json(journal,log)
                    break
    log["status"]="complete"
    publish_json(journal,log)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers",type=Path,required=True)
    p.add_argument("--lane",choices=["a","b"],required=True)
    p.add_argument("--mode",choices=["train","evaluate"],required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--sft",type=Path,required=True)
    p.add_argument("--ppo",type=Path)
    p.add_argument("--seed",type=int,default=613)
    p.add_argument("--updates",type=int,default=2,choices=range(1,9))
    p.add_argument("--horizon",type=int,default=220,choices=range(128,541))
    p.add_argument("--eval-seeds",type=int,nargs="+",default=[19111,19112])
    args=p.parse_args()
    try:
        run(args)
    except Exception as exc:
        dest=args.out/"report.json"
        if dest.exists() and not isinstance(exc,FileExistsError):
            log=json.loads(dest.read_bytes())
            log.update(status="interrupted",error=str(exc))
            publish_json(dest,log)
        raise


if __name__=="__main__":
    main()
