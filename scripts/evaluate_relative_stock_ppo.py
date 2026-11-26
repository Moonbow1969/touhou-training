"""Compare the state-only student with its online PPO descendant on uninterrupted original credits."""

import argparse
import json
from pathlib import Path

import torch

from scripts import evaluate_credit_story as story, native_story_sensors, train_relative_stock_ppo as train
from scripts.checkpoint_storage import digest
from scripts.evaluate_bomb_crossover import DATA
from scripts.evaluate_sft_native import Arm, output_lock


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--student",type=Path,default=DATA.parent/"th14-relative-phase-student-261125/student.pt")
    parser.add_argument("--risk",type=Path,default=DATA.parent/"th14-anticipatory-contact-12tick-261125/model.pt")
    parser.add_argument("--seeds",type=int,nargs="+",required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--only",choices=("frozen_student","online_ppo"))
    parser.add_argument("--menu-references",type=Path,nargs=2,required=True)
    parser.add_argument("--menu-crop",type=int,nargs=4,action="append",required=True)
    parser.add_argument("--verify-only",action="store_true")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if len(set(args.seeds))!=len(args.seeds) or min(args.seeds)<0 or len(args.menu_crop)<2:
        parser.error("Supply distinct policy seeds and verified menu crops.")
    torch.set_num_threads(1)
    args.parakit=Path(__file__).resolve().parents[2]/"parakit"
    args.target="ending";args.limit=story.LIMITS["ending"];args.start_frame=30;args.focus_wait=90
    args.crossover=[];args.timing_policy=[];args.restart_hook=None
    args.game_over_default_reference,args.game_over_retry_reference=args.menu_references
    args.menu_max_error=.035;args.transition_budget_mib=128;args.transition_reserve_mib=20480
    report=json.loads(args.checkpoint.with_name("report.json").read_text(encoding="utf-8"))
    payload=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    fingerprint=digest(args.checkpoint)
    offline=report["specification"]
    if (report["status"]!="complete" or not report["updates"] or report["updates"][-1]["sha256"]!=fingerprint
            or payload["schema"]!="relative-stock-ppo-v1" or payload["specification"]!=offline
            or offline["student_sha256"]!=digest(args.student) or offline["risk_sha256"]!=digest(args.risk)
            or payload["steps"]!=report["steps"]):
        raise ValueError("Native PPO checkpoint, configuration and final report do not match.")
    arms={}
    for name in ("frozen_student","online_ppo"):
        if args.only and args.only!=name:
            continue
        model=train.ProjectilePolicy(args.student,args.risk).eval()
        if name=="online_ppo":
            model.load_state_dict(payload["state_dict"])
        act=train.LiveActor(model,offline["temperature"],False)
        arms[name]=Arm(act,False,dict(student_sha256=offline["student_sha256"],
                                     hazard_sha256=offline["risk_sha256"],greedy=True,runtime_reference_route=False,
                                     online_steps=payload["steps"] if name=="online_ppo" else 0,
                                     checkpoint_sha256=fingerprint if name=="online_ppo" else offline["student_sha256"]))
    spec=story.run_specification(args,arms)
    spec["controller"].update(kind="pure_neural_greedy_paired_original_credit")
    spec["runtime"]["runner_sha256"]=digest(Path(__file__))
    spec["runtime"]["source_sha256"][train.__name__]=digest(Path(train.__file__))
    archive=native_story_sensors.configure(args,spec)
    if args.verify_only:
        print(json.dumps(spec["policies"]))
        return
    if not args.resume:
        args.out.mkdir(parents=True,exist_ok=False)
    menu=story.GameOverMenu(*args.menu_references,args.menu_crop,.035)
    with output_lock(args.out):
        story.evaluate(args,arms,spec,args.out/"report.json",menu,None,transition_archive=archive)


if __name__=="__main__":
    main()
