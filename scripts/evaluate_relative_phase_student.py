"""Score a frozen relative-clock neural student on stock TH14 using the standard story scorer."""
import argparse
import json
from pathlib import Path

import torch

from scripts import evaluate_credit_story as story
from scripts.checkpoint_storage import digest
from scripts.evaluate_sft_native import Arm,output_lock
from scripts.native_story_transitions import configure
from scripts.relative_phase_student import StockRelativeActor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--seeds",type=int,nargs="+",required=True)
    parser.add_argument("--target",choices=tuple(story.TARGETS),default="stage2")
    parser.add_argument("--parakit",type=Path,default=Path(__file__).resolve().parents[2]/"parakit")
    parser.add_argument("--start-frame",type=int,default=30);parser.add_argument("--focus-wait",type=int,default=90)
    parser.add_argument("--game-over-default-reference",type=Path,required=True)
    parser.add_argument("--game-over-retry-reference",type=Path,required=True)
    parser.add_argument("--menu-crop",type=int,nargs=4,action="append",required=True)
    parser.add_argument("--menu-max-error",type=float,default=.035);parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if len(set(args.seeds))!=len(args.seeds) or min(args.seeds)<0 or len(args.menu_crop)<2:
        parser.error("Use distinct nonnegative seeds and the qualified menu crops.")
    model_report=json.loads(args.weights.with_name("training.json").read_bytes())
    if model_report["hash"]!=digest(args.weights):
        parser.error("The frozen weights do not match the completed training report.")
    args.limit=story.LIMITS[args.target];args.crossover=[];args.timing_policy=[];args.restart_hook=None
    args.transition_budget_mib=128;args.transition_reserve_mib=2048
    torch.set_num_threads(1)
    arms={"relative_phase_student":Arm(StockRelativeActor(args.weights),False,
          dict(kind="state-only event-clock neural actor",sha256=model_report["hash"]))}
    spec=story.run_specification(args,arms)
    spec["controller"]["kind"]="greedy learned joint movement/focus, Shoot and Bomb"
    spec["runtime"]["source_sha256"].update({name:digest(Path(__file__).with_name(name+".py")) for name in
                                          ("relative_phase_student","state_route_student","evaluate_relative_phase_student")})
    archive=configure(args,spec)
    if not args.resume:args.out.mkdir(parents=True,exist_ok=False)
    menu=story.GameOverMenu(args.game_over_default_reference,args.game_over_retry_reference,args.menu_crop,args.menu_max_error)
    with output_lock(args.out):
        result=story.evaluate(args,arms,spec,args.out/"report.json",menu,None,transition_archive=archive)
    print(json.dumps(dict(status=result["status"])))


if __name__=="__main__":main()
