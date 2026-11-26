"""Evaluate learned Stage 2 phase labels on a fresh, uninterrupted original-game credit."""
import argparse
import json
from pathlib import Path

import torch

from scripts import evaluate_credit_story as story, native_story_sensors, stage2_phase_motor
from scripts.checkpoint_storage import digest
from scripts.evaluate_bomb_crossover import DATA
from scripts.evaluate_sft_native import Arm, output_lock
from scripts.hybrid_dodge_student import actor as avoidance
from scripts.train_full_policy import publish_json


class ProtectedStudent(stage2_phase_motor.Stage2Student):
    def __init__(self, parent, phase, protection):
        super().__init__(parent, phase)
        frozen = avoidance.StockHybridStudent(protection)
        self.contact, self.movement = frozen.contact, frozen.movement

    @torch.inference_mode()
    def predict(self, observation, deterministic=False):
        action, state = super().predict(observation, deterministic)
        joint = torch.tensor([int(action[0] + 9 * action[1])])
        x = avoidance.feature_vectors(self.contact, torch.from_numpy(observation["bullets"][None]),
            torch.from_numpy(observation["context"][None]), torch.tensor([self.environment.snapshot.stage]))
        choice = self.movement.logits(x, joint).argmax(1).item()
        action[:2] = [choice % 9, choice // 9]
        return action, state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--protection", action="store_true")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--menu-references", required=True, type=Path, nargs=2)
    p.add_argument("--menu-crop", required=True, type=int, nargs=4, action="append")
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0 or len(args.menu_crop) < 2:
        p.error("Distinct seeds and known menu locations are required.")
    torch.set_num_threads(1)
    parent = DATA.parent / "th14-relative-phase-student-261125/student.pt"
    phase = DATA.parent / "th14-state-only-stage2-phase-motor-261125/phase.pt"
    gate = json.loads(phase.with_name("offline.json").read_bytes())
    if digest(parent) != json.loads(parent.with_name("training.json").read_bytes())["hash"] or (
            digest(phase) != gate["model_sha256"] or not gate["offline_pilot_gate"]):
        raise ValueError("Stage 2 policy failed validation.")
    proof = dict(parent_sha256=digest(parent), phase_sha256=digest(phase), training_source="Alice paired training replay",
                 runtime_reference=False, runtime_collision_solver=False)
    if args.protection:
        shield = DATA.parent / "th14-hybrid-dodge-student-261125/conservative.pt"
        proof["avoidance_sha256"] = digest(shield)
        if proof["avoidance_sha256"] != "3cdd6a1501196dec9a46f07c5d708a6c0130438c7a24ed99c2e67585ee68934b":
            raise ValueError("Unexpected learned avoidance weights.")
        payload = torch.load(shield, map_location="cpu", weights_only=True)
        anchor = torch.load(parent, map_location="cpu", weights_only=True)["state_dict"]
        for k, v in anchor.items():
            torch.testing.assert_close(payload["state_dict"][k], v, rtol=0, atol=0)
        policy = ProtectedStudent(parent, phase, shield)
    else:
        policy = stage2_phase_motor.Stage2Student(parent, phase)
    arms = {"phase_motor_protected" if args.protection else "phase_motor": Arm(policy, False, proof)}
    args.parakit = Path(__file__).resolve().parents[2] / "parakit"
    args.target = "ending"
    args.limit, args.start_frame, args.focus_wait = story.LIMITS[args.target], 30, 90
    args.crossover, args.timing_policy, args.restart_hook = [], [], None
    args.game_over_default_reference, args.game_over_retry_reference = args.menu_references
    args.menu_max_error, args.transition_budget_mib, args.transition_reserve_mib = .035, 128, 20480
    spec = story.run_specification(args, arms)
    spec["controller"]["kind"] = "State-only greedy phase-conditioned learned movement, Shoot and Bomb"
    spec["runtime"]["runner_sha256"] = digest(Path(__file__))
    spec["runtime"]["source_sha256"]["phase_motor"] = digest(Path(stage2_phase_motor.__file__))
    spec["runtime"]["source_sha256"]["neural_avoidance"] = digest(Path(avoidance.__file__))
    archive = native_story_sensors.configure(args, spec)
    if args.verify_only:
        print(json.dumps(proof))
        return
    if not args.resume:
        args.out.mkdir(parents=True, exist_ok=False)
    with output_lock(args.out):
        menu = story.GameOverMenu(*args.menu_references, args.menu_crop, args.menu_max_error)
        story.evaluate(args, arms, spec, args.out/"report.json", menu, None, transition_archive=archive)
        publish_json(args.out/"phase-actions.json", dict(scored=policy.chosen, changed_vs_base=policy.changed))


if __name__ == "__main__":
    main()
