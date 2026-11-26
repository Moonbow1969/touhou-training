"""Track one uninterrupted zero-Continue Normal credit in original TH10 0.02a."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re
from shutil import disk_usage
from statistics import median
import subprocess
from time import perf_counter

from PIL import Image
from stable_baselines3 import PPO
import torch

from scripts.benchmark_trial_workers import TrialWorker
from scripts.checkpoint_storage import digest
from scripts.collect_trial_contact_windows import action_mask
from scripts.run_cross_title_th10 import SharedNative, actor
from scripts.train_full_policy import publish_json


def native(state):
    keys = ("stage", "difficulty", "lives", "score", "position", "bullet_count",
            "held", "observed_held", "native_input_bits", "time_ms")
    return {key: state[key] for key in keys}


def picture(worker, output, filename, *, live=False):
    path = output / filename
    remote = worker["capture"] + "/" + filename
    command = f'cat "{remote}"'
    if live:
        command = ('docker exec -e DISPLAY=:59 ' + worker["container"]
                   + f' scrot -z -a 0,0,640,480 /capture/{filename} && ' + command)
    result = subprocess.run([*worker["command"][:-1], command], check=True,
                            capture_output=True, timeout=9)
    data = result.stdout
    if data[:8] != b"\x89PNG\r\n\x1a\n" or len(data)>2*1024**2:
        raise RuntimeError("A native screenshot failed PNG format or size verification.")
    path.write_bytes(data)
    with Image.open(path) as im:
        if im.size!=(640,480):
            raise RuntimeError("A native screenshot differs from the game canvas.")
        im.verify()
    return dict(file=filename, sha256=digest(path), bytes=len(data))


def run(args, worker):
    if disk_usage(args.out.parent).free < 8 * 1024**3:
        raise OSError("Insufficient free evidence space.")
    if (worker["difficulty"]!=1 or not re.fullmatch(r"toho-th10-motion-story-[0-9]+",worker["container"])
            or not re.fullmatch(r"/[a-zA-Z0-9_./-]+/story-live\.[a-zA-Z0-9]+",worker["capture"])):
        raise ValueError("Use the isolated Normal story worker.")
    calibration=json.loads(args.sft.with_name("report.json").read_bytes())
    if not calibration["observed_motion_sign"] or calibration["train_titles"]!=[13,14,17]:
        raise ValueError("This protocol requires the motion-aware three-title actor.")
    torch.set_num_threads(1)
    env=SharedNative(worker,220,None)
    env.motion_sign=True
    env.count=0
    sft_sha=digest(args.sft)
    if args.ppo:
        parent=json.loads(args.ppo.with_name("report.json").read_bytes())
        checkpoint=parent["updates"][-1]
        ppo_sha=digest(args.ppo)
        if (parent["status"]!="complete" or parent["sft_sha256"]!=sft_sha
                or checkpoint["sha256"]!=ppo_sha or checkpoint["checkpoint"]!=args.ppo.name):
            raise ValueError("The frozen PPO checkpoint does not match the completed shared-SFT update.")
        model=PPO.load(args.ppo,env=env,device="cpu")
    else:
        model=actor(env,args.seed,args.sft)
        ppo_sha=None
    torch.manual_seed(args.seed)
    args.out.mkdir(exist_ok=False)
    report=dict(status="starting", title="TH10 Mountain of Faith", version="official trial 0.02a",
                host=worker["host"], difficulty="Normal", shot="Reimu A", actor_seed=args.seed,
                executable_sha256=worker["executable_sha256"], worker_sources=worker["sources"],
                runner_sha256=digest(Path(__file__)), policy_sha256=ppo_sha or sft_sha, sft_sha256=sft_sha,
                policy=("Frozen stochastic two-host live PPO from three-title SFT"
                        if ppo_sha else "Frozen stochastic replay SFT with observed motion signs"),
                train_titles=calibration["train_titles"], training_draws=calibration["train_draws"],
                game_rng_controlled=False, unpaused=worker.get("unpaused",False),
                cadence=("Continuous native clock with 85 ms nominal holds plus runner overhead"
                         if worker.get("unpaused") else "85 ms nominal resumed wall holds"),
                max_decisions=args.cap, shoot=1,bomb=0, continue_command_count=0,
                outcome=None, life_decrements=[], native_stage_entries=[], screenshots=[], errors=[],
                actor_offscreen_policy="Retain last valid actor observation during native respawn.",
                offscreen_readouts=[],hold_outliers=[])
    journal=args.out/"report.json"
    path=args.out/"credit.jsonl.gz"
    publish_json(journal,report)
    action_counts=Counter()
    timing=[]
    native_matches=0
    movements=0
    prior=None
    started=perf_counter()

    def save_picture(filename,*,live=False,decision=None):
        try:
            row=picture(worker,args.out,filename,live=live)
        except (OSError,RuntimeError,subprocess.SubprocessError) as error:
            report["errors"].append(dict(capture=filename,type=type(error).__name__))
        else:
            report["screenshots"].append(dict(decision=decision,**row))

    try:
        with TrialWorker(worker["command"]) as pipe, gzip.open(path,"wt",encoding="utf-8") as trace:
            prior=pipe.read(timeout=85)
            if (prior["stage"],prior["difficulty"],prior["lives"],prior["score"],prior["held"])!=(1,1,2,0,0):
                raise RuntimeError("Native opening stock, rank, score or stage failed qualification.")
            report["opening"]=native(prior)
            report["native_stage_entries"].append(dict(decision=0,**native(prior)))
            report["status"]="running"
            publish_json(journal,report)
            save_picture("decision-0000.png",live=True,decision=0)
            obs=env.observe(prior,prior,0)
            previous_active=prior
            play_start=perf_counter()
            for index in range(1,args.cap+1):
                action,_=model.predict(obs,deterministic=False)
                move,focus=map(int,action)
                mask=action_mask(move,focus)
                state=pipe.request(f"S {mask}")
                if state["held"]!=mask or state["difficulty"]!=1:
                    raise RuntimeError("The native bridge action or assigned difficulty changed.")
                hold=state["time_ms"]
                if not 70<=hold<=270:
                    report["hold_outliers"].append(dict(decision=index,measured_ms=hold))
                    if not worker.get("unpaused") or not 0<=hold<=2500:
                        raise RuntimeError("A measured native command exceeded the hold tolerance.")
                before=native(prior)
                after=native(state)
                timing.append(state["time_ms"])
                native_matches+=int(state["observed_held"]==mask)
                movements+=int(state["position"]!=prior["position"])
                action_counts[(move,focus)]+=1
                lost=state["lives"]<prior["lives"]
                advanced=state["stage"]>prior["stage"]
                if lost:
                    report["life_decrements"].append(dict(decision=index,before=prior["lives"],**after))
                if advanced:
                    report["native_stage_entries"].append(dict(decision=index,**after))
                if state["stage"]<prior["stage"]:
                    raise RuntimeError("The native stage regressed within an uninterrupted credit.")
                if 0<=state["position"][1]<=450:
                    observation_previous=prior if prior is previous_active else state
                    obs=env.observe(state,observation_previous,focus)
                    previous_active=state
                else:
                    report["offscreen_readouts"].append(dict(decision=index,**after))
                prior=state
                trace.write(json.dumps(dict(decision=index, action=[move,focus,1,0],
                    requested=mask, native_before=before, native_after=after))+"\n")
                if lost or advanced or index % args.screenshot_interval == 0:
                    save_picture(f"decision-{index:04}.png",live=True,decision=index)
                    trace.flush()
                    report["last_acknowledged"]=dict(decision=index,**after)
                    publish_json(journal,report)
                if advanced:
                    report["outcome"]="native_stage2_entered"
                    break
                if state["lives"]<0:
                    report["outcome"]="negative_lives_awaiting_visual_verification"
                    break
            else:
                report["outcome"]="budget_censored"
            report["startup_to_endpoint_control_seconds"]=perf_counter()-play_start
    except (OSError,RuntimeError,ValueError) as error:
        report["outcome"]="native_stream_censored_awaiting_visual_verification"
        report["errors"].append(dict(type=type(error).__name__,message=str(error)))
    for number in range(1,4):
        save_picture(f"settled-{number}.png")
    report.update(status="complete", decisions=len(timing), game_resumed_seconds=sum(timing)/1000,
                  median_hold_ms=median(timing) if timing else None, wall_seconds=perf_counter()-started,
                  trace_sha256=digest(path) if path.exists() else None, native_inputs_matched=native_matches,
                  actual_motion_decisions=movements, final_native=native(prior) if prior else None,
                  move_focus_counts=[[action_counts[(d,f)] for d in range(9)] for f in range(2)])
    publish_json(journal,report)
    print(json.dumps({key:report[key] for key in ("outcome","decisions","game_resumed_seconds",
        "native_inputs_matched","actual_motion_decisions","life_decrements","native_stage_entries")}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",type=Path,required=True)
    parser.add_argument("--sft",type=Path,required=True)
    parser.add_argument("--ppo",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=19611)
    parser.add_argument("--cap",type=int,default=1800,choices=range(1000,2401),metavar="1000–2400")
    parser.add_argument("--screenshot-interval",type=int,default=160,choices=range(100,301),metavar="100–300")
    args=parser.parse_args()
    run(args,json.loads(args.worker.read_bytes()))


if __name__=="__main__":
    main()
