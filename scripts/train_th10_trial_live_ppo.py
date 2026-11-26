"""Small on-policy TH10 0.02a live PPO pilot with a matched frozen initialization."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
from shutil import disk_usage
from time import perf_counter

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import torch

from scripts.benchmark_trial_workers import TrialWorker
from scripts.checkpoint_storage import digest
from scripts.collect_trial_contact_windows import action_mask
from scripts.train_full_policy import publish_json
from scripts.train_structured import BulletSetEncoder


BINARY = "5430fac601d645798ccf794362ec7d4cd753814f09d160247dc628b7c6943fd8"
ARMS = ("frozen_init", "live_ppo")
ENCODING = (384., 448., 16., 16.)


class NativeTH10(gym.Env):
    observation_space = gym.spaces.Dict({
        "bullets": gym.spaces.Box(-1,1,(96,7),dtype=np.float32),
        "context": gym.spaces.Box(-1,1,(10,),dtype=np.float32),
    })
    action_space = gym.spaces.MultiDiscrete((9,2))

    def __init__(self, worker, horizon, trace):
        if worker["executable_sha256"] != BINARY:
            raise ValueError("This adapter requires the same official TH10 trial.")
        self.worker, self.horizon, self.trace = worker, horizon, trace
        self.difficulty = worker.get("difficulty",1)
        self.pipe = None
        self.episode = 0
        self.executed = 0
        self.trace_tag = {}

    def close(self):
        if self.pipe is not None:
            pipe, self.pipe = self.pipe, None
            pipe.__exit__(None,None,None)

    def observe(self, state, previous, focus):
        count = state["bullet_count"]
        if not 0 <= count <= 2001 or len(state["bullets"]) != min(count,96):
            raise ValueError("The official trial produced inconsistent projectile counts.")
        bullets = np.zeros((96,7),dtype=np.float32)
        if count:
            n = min(96,count)
            bullets[:n,:4] = np.asarray(state["bullets"],dtype=np.float32) / ENCODING
            bullets[:n,5:] = 1
        x,y = state["position"]
        px,py = previous["position"]
        context = np.array((x/192, y/448, state["lives"]/8, state["score"]/20000, count/512,
                            focus, self.count/self.horizon, (x-px)/32, (y-py)/32, 1.),dtype=np.float32)
        if not np.isfinite(bullets).all() or not np.isfinite(context).all():
            raise ValueError("The original trial state is not finite.")
        return dict(bullets=bullets.clip(-1,1),context=context.clip(-1,1))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        started=perf_counter()
        reused=False
        fallback=None
        if self.pipe is not None and self.worker.get("native_retry"):
            try:
                state=self.pipe.request("R")
                if (state["stage"],state["lives"],state["score"],state["held"])!=(1,2,0,0):
                    raise RuntimeError("Warm retry did not restore the initial stock and score.")
                x,y=state["position"]
                if abs(x)>.1 or abs(y-400)>1 or state.get("difficulty",1)!=self.difficulty:
                    raise RuntimeError("Warm retry did not restore the expected spawn and difficulty.")
                self.state=state
                reused=True
            except (RuntimeError,OSError,EOFError) as exc:
                fallback=str(exc)
        if not reused:
            self.close()
            self.pipe = TrialWorker(self.worker["command"], shutdown_timeout=24).__enter__()
            self.state = self.pipe.read(timeout=84)
        if (self.state["stage"], self.state["lives"], self.state["score"], self.state["held"]) != (1,2,0,0):
            raise RuntimeError("Native TH10 did not start with untouched Stage 1 inventory.")
        if self.state.get("difficulty",1) != self.difficulty:
            raise RuntimeError("The verified TH10 difficulty differs from the assigned test.")
        self.episode += 1
        self.count = self.elapsed = self.horizontal = self.responses = self.bullet_peak = 0
        self.actions = Counter()
        return self.observe(self.state,self.state,0), dict(resources=self.state["lives"],stock_score=self.state["score"],
            native_retry=reused,reset_seconds=perf_counter()-started,reset_fallback=fallback)

    def step(self, action):
        move, focus = map(int,action)
        mask = action_mask(move,focus)
        before = self.state
        state = self.pipe.request(f"S {mask}")
        self.state = state
        if state["held"] != mask or not 70 <= state["time_ms"] <= 270:
            raise RuntimeError("The native input-held acknowledgement or measured cadence changed.")
        if state.get("difficulty",1) != self.difficulty:
            raise RuntimeError("The verified TH10 difficulty changed inside the assigned test.")
        self.count += 1
        self.actions[(move,focus)] += 1
        self.elapsed += state["time_ms"]
        miss = state["lives"] < before["lives"]
        advanced = state["stage"] > 1
        terminated = miss or advanced
        truncated = self.count >= self.horizon and not terminated
        self.bullet_peak = max(self.bullet_peak,state["bullet_count"])
        direction = move%3 - 1
        x = before["position"][0]
        if not terminated and direction and -165 < x < 165:
            self.horizontal += 1
            self.responses += direction * (state["position"][0] - x) > .05
        reward = (0 if miss else 1./self.horizon) - float(miss)
        raw_trace = dict(credit=self.episode,decision=self.count,action=[move,focus,1,0],ack=state["held"],
                         native_difficulty=self.difficulty,
                         native_before={key:before[key] for key in ("stage","lives","score","position","bullet_count")},
                         native_after={key:state[key] for key in ("stage","lives","score","position","bullet_count","time_ms")},
                         terminated=terminated,truncated=truncated,reward=reward, **self.trace_tag)
        self.trace.write(json.dumps(raw_trace)+"\n")
        self.executed += 1
        if self.count % 64 == 0 or terminated or truncated:
            self.trace.flush()
        info = dict(outcome="first_miss" if miss else "stage_advanced" if advanced else "horizon" if truncated else None,
                    native_stage=state["stage"],remaining_lives=state["lives"],raw_score=state["score"],
                    decisions=self.count, resumed_wall_ms=self.elapsed,first_miss=miss, peak_bullets=self.bullet_peak,
                    horizontal_requested=self.horizontal,horizontal_responses=int(self.responses),
                    move_focus_counts=[[self.actions[(d,f)] for d in range(9)] for f in range(2)])
        if terminated or truncated:
            print(json.dumps(dict(event="native_credit",**info)),flush=True)
        return self.observe(state,before,focus), reward, terminated, truncated, info


def record(args, workers):
    torch.set_num_threads(1)
    if disk_usage(args.out.parent).free < 8 * (1024**3):
        raise OSError("Insufficient reserve on the evidence volume.")
    args.out.mkdir(parents=False,exist_ok=False)
    report = dict(status="starting", title="TH10 Mountain of Faith", version="official trial 0.02a",
                  executable_sha256=BINARY, setting="Normal Reimu A first-life pilot", shoot=1,bomb=0,
                  training_steps=args.steps, horizon=args.horizon, controlled_rng="Actor only", seed=args.seed,
                  eval_seeds=args.eval_seeds, train_host="accelerator", eval_host="entropy",
                  observation="Raw position, life stock, score and active projectile positions and velocities; no radius used.",
                  cadence="Own game suspended during inference, resumed for nominal 85 ms per action. Actual milliseconds logged.",
                  task="Native first-life survival, using a restricted decision budget. No stage-clear claim.",
                  reward="1/horizon per non-injury action, minus 1 at actual native life decrement.",
                  runner_sha256=digest(Path(__file__)),
                  workers={name:{key:row[key] for key in ("executable_sha256","runtime","sources")} for name,row in workers.items()})
    path = args.out/"report.json"
    publish_json(path,report)
    with gzip.open(args.out/"train-transitions.jsonl.gz","wt",encoding="utf-8") as trace:
        env = NativeTH10(workers["accelerator"],args.horizon,trace)
        try:
            actor = PPO("MultiInputPolicy",env,seed=args.seed,device="cpu",n_steps=128,batch_size=64,n_epochs=4,
                        gamma=.996,gae_lambda=.95,learning_rate=.00025,ent_coef=.012,target_kl=.03,
                        policy_kwargs=dict(features_extractor_class=BulletSetEncoder,net_arch=dict(pi=[64,64],vf=[64,64])))
            initial={key:value.detach().clone() for key,value in actor.policy.state_dict().items()}
            actor.save(args.out/"frozen_init")
            report.update(status="training",frozen_init_sha256=digest(args.out/"frozen_init.zip"),
                          training_updates=[], discarded_rollouts=[],
                          train_parameters=sum(parameter.numel() for parameter in actor.policy.parameters()))
            publish_json(path,report)
            start=perf_counter()
            checkpoint = args.out/"frozen_init.zip"
            for update in range(1,args.steps//128+1):
                for attempt in range(1,4):
                    before = env.executed
                    env.trace_tag = dict(update=update,attempt=attempt)
                    try:
                        actor.learn(128,reset_num_timesteps=False)
                    except (RuntimeError,OSError,ValueError) as error:
                        report["discarded_rollouts"].append(dict(update=update,attempt=attempt,
                            executed_but_unused=env.executed-before,error=str(error)))
                        publish_json(path,report)
                        env.close()
                        if attempt==3:
                            raise
                        actor = PPO.load(checkpoint,env=env,device="cpu",force_reset=True)
                    else:
                        if actor.num_timesteps != 128*update or env.executed-before != 128:
                            raise RuntimeError("The committed PPO step count does not match acknowledged actions.")
                        prepared = args.out/"committed-next.zip"
                        actor.save(prepared)
                        checkpoint = args.out/"committed.zip"
                        prepared.replace(checkpoint)
                        report["training_updates"].append(dict(update=update,attempt=attempt,
                                                                 actual_steps=actor.num_timesteps,
                                                                 checkpoint_sha256=digest(checkpoint)))
                        publish_json(path,report)
                        print(json.dumps(dict(event="ppo_update_committed",**report["training_updates"][-1])),flush=True)
                        break
            actor.save(args.out/"live_ppo")
            report.update(status="evaluating",actual_steps=actor.num_timesteps,training_seconds=perf_counter()-start,
                          live_ppo_sha256=digest(args.out/"live_ppo.zip"),
                          policy_weights_l2_change=float(torch.sqrt(sum((value-initial[name]).square().sum()
                              for name,value in actor.policy.state_dict().items())).item()))
            publish_json(path,report)
        finally:
            env.close()
    models={name:PPO.load(args.out/f"{name}.zip",device="cpu") for name in ARMS}
    report["trials"], report["discarded_evaluations"] = [], []
    for block, seed in enumerate(args.eval_seeds):
        for offset in range(2):
            arm=ARMS[(block+offset)%2]
            for attempt in range(1,4):
                trace_path=args.out/f"eval-{seed}-{arm}-attempt{attempt}.jsonl.gz"
                with gzip.open(trace_path,"wt",encoding="utf-8") as trace:
                    env=NativeTH10(workers["entropy"],args.horizon,trace)
                    torch.manual_seed(seed)
                    try:
                        observation,_=env.reset()
                        ended=False
                        while not ended:
                            action,_=models[arm].predict(observation,deterministic=False)
                            observation,_,terminated,truncated,info=env.step(action)
                            ended=terminated or truncated
                    except (RuntimeError,OSError,ValueError) as error:
                        report["discarded_evaluations"].append(dict(arm=arm,seed=seed,attempt=attempt,
                                                                   acknowledged=env.executed,error=str(error),
                                                                   trace=trace_path.name))
                        publish_json(path,report)
                        if attempt==3:
                            raise
                    else:
                        report["trials"].append(dict(arm=arm,seed=seed,order=offset+1,attempt=attempt,
                                                     trace=trace_path.name,**info))
                        publish_json(path,report)
                        break
                    finally:
                        env.close()
    report["summary"]={}
    for arm in ARMS:
        rows=[row for row in report["trials"] if row["arm"]==arm]
        report["summary"][arm]=dict(trials=len(rows),first_misses=sum(row["first_miss"] for row in rows),
                                   median_decisions=float(np.median([row["decisions"] for row in rows])),
                                   median_resumed_ms=float(np.median([row["resumed_wall_ms"] for row in rows])),
                                   median_score=float(np.median([row["raw_score"] for row in rows])),
                                   max_bullets=max(row["peak_bullets"] for row in rows))
    report["status"]="complete"
    publish_json(path,report)
    print(json.dumps(report["summary"],indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--steps",type=int,default=512)
    parser.add_argument("--horizon",type=int,default=256)
    parser.add_argument("--seed",type=int,default=101807)
    parser.add_argument("--eval-seeds",type=int,nargs="+",default=[6101,6102,6103])
    args=parser.parse_args()
    if (args.steps%128 or not 128 <= args.steps <= 4096 or not 128 <= args.horizon <= 540 or
            len(args.eval_seeds)<2 or len(set(args.eval_seeds))!=len(args.eval_seeds) or args.seed in args.eval_seeds):
        parser.error("Use a bounded PPO budget and distinct independent external seeds.")
    record(args,json.loads(args.workers.read_text()))


if __name__=="__main__":
    main()
