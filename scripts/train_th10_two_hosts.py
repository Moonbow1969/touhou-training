"""Synchronous online PPO with two physical TH10 trial workers and a replay-trained actor."""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait
from contextlib import ExitStack
from copy import deepcopy
import gzip
import json
from pathlib import Path
from shutil import disk_usage
from time import perf_counter

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from scripts.checkpoint_storage import digest
from scripts.run_cross_title_th10 import SharedNative,actor
from scripts.train_full_policy import publish_json


class ParallelNative(DummyVecEnv):
    """Run independent blocked native SSH requests concurrently and retain SB3 reset semantics."""
    def __init__(self, factories):
        super().__init__(factories)
        self.pool=ThreadPoolExecutor(max_workers=len(factories))

    def gather(self,fn):
        futures=[self.pool.submit(fn,i) for i in range(self.num_envs)]
        wait(futures)
        return [f.result() for f in futures]

    def reset(self):
        results=self.gather(lambda i:self.envs[i].reset(seed=self._seeds[i],options=self._options[i]))
        for i,(obs,info) in enumerate(results):
            self.reset_infos[i]=info
            self._save_obs(i,obs)
        self._reset_seeds()
        self._reset_options()
        return self._obs_from_buf()

    def step_wait(self):
        def step(i):
            obs,reward,term,trunc,info=self.envs[i].step(self.actions[i])
            info["TimeLimit.truncated"]=trunc and not term
            if term or trunc:
                info["terminal_observation"]=obs
                obs,self.reset_infos[i]=self.envs[i].reset()
            return obs,reward,term or trunc,info
        for i,(obs,reward,done,info) in enumerate(self.gather(step)):
            self.buf_rews[i]=reward
            self.buf_dones[i]=done
            self.buf_infos[i]=info
            self._save_obs(i,obs)
        return self._obs_from_buf(),self.buf_rews.copy(),self.buf_dones.copy(),deepcopy(self.buf_infos)

    def close(self):
        try:
            self.gather(lambda i:self.envs[i].close())
        finally:
            self.pool.shutdown()


class MeasuredGame(SharedNative):
    def reset(self,**kwargs):
        obs,info=super().reset(**kwargs)
        self.reset_records.append(info)
        return obs,info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers",type=Path,required=True)
    p.add_argument("--sft",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--updates",type=int,default=8,choices=range(1,17))
    p.add_argument("--horizon",type=int,default=220)
    p.add_argument("--seed",type=int,default=19401)
    args=p.parse_args()
    workers=json.loads(args.workers.read_bytes())
    if len(workers) not in (2,4) or disk_usage(args.out.parent).free<8*1024**3:
        raise ValueError("Expected two or four independent workers and 8 GiB of free storage.")
    rollout_steps=128*len(workers)
    args.out.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    journal=args.out/"report.json"
    report=dict(status="training",source_sha256=digest(Path(__file__)),hosts=list(workers),seed=args.seed,
                sft_sha256=digest(args.sft),steps_per_actor_per_update=128,updates=[],failed_attempts=[],
                observation="Shared motion-sign structured sensor, human model fitted to TH13, TH14 and TH17.",
                reward="First spare-life loss minus one, plus 1/horizon per confirmed surviving action.",
                horizon=args.horizon,requested_updates=args.updates,
                assignments={lane:dict(host=w.get("host",lane),difficulty=w.get("difficulty",1),
                    native_retry=w.get("native_retry",False),worker_sources=w["sources"]) for lane,w in workers.items()})
    publish_json(journal,report)
    checkpoint=None
    try:
        with ExitStack() as stack:
            games=[]
            for host,worker in workers.items():
                trace=stack.enter_context(gzip.open(args.out/f"{host}.jsonl.gz","wt",encoding="utf-8"))
                game=MeasuredGame(worker,args.horizon,trace)
                game.reset_records=[]
                game.motion_sign=True
                games.append(game)
            env=ParallelNative([lambda game=game:game for game in games])
            stack.callback(env.close)
            model=actor(env,args.seed,args.sft)
            if not json.loads(args.sft.with_name("report.json").read_bytes()).get("observed_motion_sign"):
                raise ValueError("Training and deployment motion features differ.")
            checkpoint=args.out/"initial.zip"
            model.save(checkpoint)
            for update in range(1,args.updates+1):
                for attempt in range(1,4):
                    start=perf_counter()
                    before=[g.executed for g in games]
                    for game in games:
                        game.trace_tag=dict(update=update,attempt=attempt)
                    try:
                        model.learn(rollout_steps,reset_num_timesteps=False)
                    except (RuntimeError,OSError,TimeoutError,EOFError,ValueError) as exc:
                        steps=[g.executed-n for g,n in zip(games,before)]
                        report["failed_attempts"].append(dict(update=update,attempt=attempt,discarded_by_host=steps,
                                                               seconds=perf_counter()-start,reason=str(exc)[:300]))
                        publish_json(journal,report)
                        env.gather(lambda i:games[i].close())
                        if attempt==3:
                            raise
                        model=PPO.load(checkpoint,env=env,force_reset=True,device="cpu")
                        continue
                    acknowledged=[g.executed-n for g,n in zip(games,before)]
                    if acknowledged!=[128]*len(workers) or model.num_timesteps!=rollout_steps*update:
                        raise ValueError("Per-host acknowledged actions disagree with the on-policy buffer.")
                    checkpoint=args.out/f"update-{update}.zip"
                    model.save(checkpoint)
                    elapsed=perf_counter()-start
                    report["updates"].append(dict(update=update,attempt=attempt,acknowledged=acknowledged,
                        accepted_total=model.num_timesteps,seconds=elapsed,accepted_decisions_per_second=rollout_steps/elapsed,
                        checkpoint=checkpoint.name,sha256=digest(checkpoint)))
                    report["resets"]={key:g.reset_records for key,g in zip(workers,games)}
                    publish_json(journal,report)
                    break
        report["status"]="complete"
    except Exception as exc:
        report.update(status="interrupted",error=str(exc)[:300])
        raise
    finally:
        publish_json(journal,report)


if __name__=="__main__":
    main()
