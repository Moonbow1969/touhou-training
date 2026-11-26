"""Use a frozen learned 12-tick contact prediction to rerank a demonstration tracker's movement."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from scripts import evaluate_credit_story as story, evaluate_demo_route_encounters as encounter
from scripts import evaluate_demo_route_stage2_phase as phase2, native_story_sensors, train_anticipatory_contact as risk
from scripts.checkpoint_storage import digest
from scripts.evaluate_bomb_crossover import DATA, TEMPLATE
from scripts.evaluate_sft_native import load_arm, output_lock
from scripts.native_story_restart import GameOverMenu
from scripts.train_full_policy import publish_json


class RiskRoute(phase2.ExactStage2Route):
    def enable_contact(self, predictor, strength, stages):
        self.predictor = predictor
        self.strength = strength
        self.stages = frozenset(stages)
        def capture(_layer, _inputs, output):
            self.last_logits = output[0]
        self.model.register_forward_hook(capture)

    @torch.inference_mode()
    def predict(self, observation, deterministic=False):
        action, state = super().predict(observation, deterministic)
        env = self.environment
        if (env.snapshot.stage not in self.stages or env.snapshot.player_state != 1
                or action[3] or "x" in env.held or env.game.read_game_val("in_dialog") == -1):
            return action, state
        focus = int(action[1])
        logits = self.last_logits.reshape(2,9)[focus]
        if int(logits.argmax()) != action[0]:
            raise RuntimeError("Recorded original movement disagrees with the frozen route tracker.")
        ctx = torch.from_numpy(observation["context"][None])
        bullets = torch.from_numpy(observation["bullets"][None])
        costs = self.predictor(bullets,risk.local.local_context(ctx,0))[0,:18].reshape(2,9)[focus].sigmoid()
        alternative = int((logits-self.strength*costs).argmax())
        summary = self.sessions[-1].setdefault("anticipatory_contact", {}).setdefault(
            str(env.snapshot.stage), dict(eligible=0,changed=0,max_estimated_drop=0.))
        summary["eligible"] += 1
        if alternative != action[0]:
            summary["changed"] += 1
            summary["max_estimated_drop"] = max(summary["max_estimated_drop"],float(costs[action[0]]-costs[alternative]))
            action = action.copy()
            action[0] = alternative
        return action, state


def build(args):
    parent = encounter.make_arms(args)["macro_encounter_clock"]
    report = json.loads((args.artifact/"report.json").read_text(encoding="utf-8"))
    route = args.artifact/"route.npz"
    import numpy as np
    with np.load(route,allow_pickle=False) as z:
        plan,_ = phase2.stage2_knots(args.data/"th14_ud0613-stage2.npz",report["train_stage_shards"]["2"],z["frame_2"])
    shoot = load_arm(args.shoot_policy)
    bomb = json.loads(args.bomb_model.read_text(encoding="utf-8"))
    model = risk.AnticipatoryContact().eval()
    evidence = json.loads(args.contact.with_name("report.json").read_text(encoding="utf-8"))
    if (evidence["status"]!="complete" or digest(args.contact)!=evidence["model_sha256"]
            or evidence["horizon_native_ticks"]!=12 or evidence["source_sha256"]!=digest(Path(risk.__file__))):
        raise ValueError("Mismatched anticipatory predictor or training implementation.")
    ckpt = torch.load(args.contact,map_location="cpu",weights_only=True)
    if ckpt["schema"]!="th14-learned-contact-12tick-v1":
        raise ValueError("Incompatible contact model.")
    model.load_state_dict(ckpt["state_dict"])
    args.actor_sources = dict(anticipatory_model_sha256=evidence["model_sha256"], stage2_exact_knots=plan,
                              adjust_stages=args.stages, contact_penalty=args.strength,
                              train_target="Projected 12-tick held-command contact on recorded exogenous fields.")
    base = (args.artifact/"model.pt",route,bomb,shoot,plan)
    result = {}
    if args.paired:
        anchor = phase2.ExactStage2Route(*base)
        anchor.knots = parent.policy.knots
        result["exact_reference"] = replace(parent, policy=anchor, provenance=dict(parent.provenance, stage2_exact_knots=plan))
    actor = RiskRoute(*base)
    actor.knots = parent.policy.knots
    actor.enable_contact(model,args.strength,args.stages)
    result["exact_with_12tick_contact"] = replace(parent,policy=actor,provenance=dict(parent.provenance,**args.actor_sources))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact",type=Path,default=DATA/"th14-demo-alice-route-261125")
    parser.add_argument("--data",type=Path,default=DATA/"sft-native-data-261123/frozen-training-snapshot")
    parser.add_argument("--contact",type=Path,default=DATA.parent/"th14-anticipatory-contact-12tick-261125/model.pt")
    parser.add_argument("--bomb-model",type=Path,default=DATA.parent/"th14-randomized-bomb-value-261125/bomb_value.json")
    parser.add_argument("--strength",type=float,choices=(8.,16.,32.,64.),required=True)
    parser.add_argument("--stages",nargs="+",type=int,choices=range(1,7),default=[1,2])
    parser.add_argument("--seeds",nargs="+",type=int,required=True)
    parser.add_argument("--paired",action="store_true")
    parser.add_argument("--verify-only",action="store_true")
    parser.add_argument("--out",type=Path)
    parser.add_argument("--menu-references",nargs=2,type=Path)
    parser.add_argument("--menu-crop",action="append",nargs=4,type=int)
    parser.add_argument("--resume",action="store_true")
    parser.set_defaults(shoot_policy=TEMPLATE,compare_reference=False,parakit=Path(phase2.__file__).resolve().parents[2]/"parakit",
        target="ending",limit=story.LIMITS["ending"],start_frame=30,focus_wait=90,
        game_over_default_reference=None,game_over_retry_reference=None,menu_max_error=.035,
        transition_budget_mib=96,transition_reserve_mib=20480,crossover=[],timing_policy=[],restart_hook=None)
    args=parser.parse_args()
    if (len(set(args.stages))!=len(args.stages) or len(set(args.seeds))!=len(args.seeds) or min(args.seeds)<0
            or not args.verify_only and args.out is None or bool(args.menu_references)!=bool(args.menu_crop)
            or args.menu_crop and len(args.menu_crop)<2):
        parser.error("Supply distinct seeds/stages and both menu templates with their crops.")
    torch.set_num_threads(1)
    if args.menu_references:
        args.game_over_default_reference,args.game_over_retry_reference=args.menu_references
    arms=build(args)
    spec=story.run_specification(args,arms)
    spec["controller"].update(kind="demo_route_with_learned_12_tick_contact", separate_focus_shoot_bomb_unchanged=True)
    spec["runtime"]["runner_sha256"]=digest(Path(__file__))
    for mod in (risk,encounter,phase2,story):
        spec["runtime"]["source_sha256"][mod.__name__]=digest(Path(mod.__file__))
    archive=native_story_sensors.configure(args,spec)
    if args.verify_only:
        print(json.dumps(spec["policies"]))
        return
    if not args.resume:
        args.out.mkdir(parents=True,exist_ok=False)
    menu=GameOverMenu(*args.menu_references,args.menu_crop,.035) if args.menu_references else None
    with output_lock(args.out):
        try:
            story.evaluate(args,arms,spec,args.out/"report.json",menu,None,transition_archive=archive)
        finally:
            path=args.out/"controller-contact-telemetry.json"
            prior=json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
            prior.append({name:arm.policy.sessions for name,arm in arms.items()})
            publish_json(path,prior)

if __name__=="__main__":
    main()
