"""Replace only the state-only student's Shoot head with the frozen human SFT decoder on every decision."""

import argparse
import json
from pathlib import Path

import torch

from scripts import evaluate_credit_story as story, native_story_sensors
from scripts.checkpoint_storage import digest
from scripts.evaluate_bomb_crossover import DATA, TEMPLATE
from scripts.evaluate_sft_native import Arm, load_arm, output_lock
from scripts.evaluate_stage_policies import policy_observation
from scripts.pretrain_policy import predict_heads
from scripts.relative_phase_student import StockRelativeActor
from scripts.train_full_policy import publish_json


class IndependentShoot(StockRelativeActor):
    def __init__(self,student,shoot):
        super().__init__(student)
        self.shoot=shoot
        self.motion_spans=shoot.policy.motion_spans
        self.retain_recent_motion=shoot.policy.retain_recent_motion
        self.tallies={}

    @torch.inference_mode()
    def predict(self,observation,deterministic=False):
        action,state=super().predict(observation,deterministic)
        env=self.environment
        transformed=policy_observation(self.shoot.policy,env,observation,uses_history=self.shoot.history)
        tensors,_=self.shoot.policy.obs_to_tensor(transformed)
        logits=predict_heads(self.shoot.policy,tensors["bullets"],tensors["context"])[2]
        fired=int(logits.argmax(1).item())
        key=f"credit_{env.reset_count}_stage_{env.snapshot.stage}"
        count=self.tallies.setdefault(key,dict(decisions=0,base_firing=0,transferred_firing=0,added=0,removed=0))
        count["decisions"]+=1
        count["base_firing"]+=int(action[2])
        count["transferred_firing"]+=fired
        count["added"]+=int(fired and not action[2])
        count["removed"]+=int(not fired and action[2])
        action[2]=fired
        return action,state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student",type=Path,default=DATA.parent/"th14-relative-phase-student-261125/student.pt")
    parser.add_argument("--shoot",type=Path,default=TEMPLATE)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--seeds",nargs="+",type=int,required=True)
    parser.add_argument("--menu-references",nargs=2,type=Path,required=True)
    parser.add_argument("--menu-crop",nargs=4,type=int,action="append",required=True)
    parser.add_argument("--paired",action="store_true")
    parser.add_argument("--verify-only",action="store_true")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if not args.seeds or min(args.seeds)<0 or len(set(args.seeds))!=len(args.seeds) or len(args.menu_crop)<2:
        parser.error("Supply distinct seeds and known menu locations.")
    if digest(args.student)!=json.loads(args.student.with_name("training.json").read_bytes())["hash"]:
        raise ValueError("The state-only checkpoint and its training report disagree.")
    torch.set_num_threads(1)
    shoot=load_arm(args.shoot)
    policy=IndependentShoot(args.student,shoot)
    arms={}
    if args.paired:
        arms["state_only_original"]=Arm(StockRelativeActor(args.student),False,dict(student_sha256=digest(args.student)))
    arms["state_only_sft_shoot"]=Arm(policy,shoot.history,dict(student_sha256=digest(args.student),
                                     shoot=shoot.provenance,applied_heads=["shoot"],runtime_trajectory=False))
    args.target="ending";args.limit=story.LIMITS[args.target];args.start_frame=30;args.focus_wait=90
    args.parakit=Path(__file__).resolve().parents[2]/"parakit"
    args.crossover=[];args.timing_policy=[];args.restart_hook=None
    args.game_over_default_reference,args.game_over_retry_reference=args.menu_references
    args.menu_max_error=.035;args.transition_budget_mib=128;args.transition_reserve_mib=20480
    spec=story.run_specification(args,arms)
    spec["controller"]["kind"]="state-only student with unclamped learned SFT Shoot on every decision"
    spec["runtime"]["runner_sha256"]=digest(Path(__file__))
    archive=native_story_sensors.configure(args,spec)
    if args.verify_only:
        print(json.dumps(spec["policies"]))
        return
    if not args.resume:
        args.out.mkdir(parents=True,exist_ok=False)
    menu=story.GameOverMenu(*args.menu_references,args.menu_crop,.035)
    with output_lock(args.out):
        try:
            story.evaluate(args,arms,spec,args.out/"report.json",menu,None,transition_archive=archive)
        finally:
            p=args.out/"shoot-decisions.json"
            previous=json.loads(p.read_text(encoding="utf-8")) if p.is_file() else []
            previous.append(policy.tallies)
            publish_json(p,previous)

if __name__=="__main__":
    main()
