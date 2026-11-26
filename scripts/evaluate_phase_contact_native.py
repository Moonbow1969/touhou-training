"""Compare the frozen Alice phase tracker with and without a neural contact penalty."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np
import torch

from scripts import evaluate_credit_story as story, evaluate_demo_route_phase as phase
from scripts import evaluate_phase_contact as audit, fit_randomized_bomb_value, native_story_transitions
from scripts import train_demo_route, train_local_avoidance, train_native_bomb_timing
from scripts.checkpoint_storage import digest
from scripts.evaluate_bomb_crossover import TEMPLATE
from scripts.evaluate_sft_native import Arm, load_arm, output_lock
from scripts.native_story_restart import GameOverMenu
from scripts.train_full_policy import publish_json


PENALTY = 64.
DEFAULT_BOMB = audit.local.DATA.parent / "th14-randomized-bomb-value-261125/bomb_value.json"
DEFAULT_AUDIT = audit.local.DATA.parent / "th14-phase-contact-counterfactual-2201-261125"
SHOOT_FINGERPRINT = "d62aea41897019df9bec90f3f3288a350a0dfbd1bd471fbee0377400cdc1e2bc"
BOMB_FINGERPRINT = "51f89c4f816dc89f6bfd0b3e672625b1a4a6c116f0a01cdb3759332cad0b81f8"


class ContactPhaseRoute(phase.PhaseRoute):
    def __init__(self, checkpoint, route, bomb, shoot, contact):
        super().__init__(checkpoint, route, bomb, shoot, "phase-aware")
        self.contact = contact
        self._tracker_logits = None
        self.model.register_forward_hook(self.capture_tracker)

    def capture_tracker(self, _network, _inputs, output):
        self._tracker_logits = output[0].reshape(1, 18)

    @torch.inference_mode()
    def predict(self, observation, deterministic=False):
        self._tracker_logits = None
        action, state = super().predict(observation, deterministic)
        env = self.environment
        if (env.snapshot.stage != 1 or env.snapshot.player_state != 1
                or action[3] or "x" in env.held):
            return action, state
        if self._tracker_logits is None:
            raise RuntimeError("The base Alice tracker did not provide its movement logits.")
        context = torch.from_numpy(observation["context"][None])
        bullets = torch.from_numpy(observation["bullets"][None])
        danger = self.contact(bullets, train_local_avoidance.local_context(context, 0))[0].sigmoid()
        proposed, focus, old, scores = audit.rerank(self._tracker_logits, danger, PENALTY)
        move, original = int(proposed.item()), int(old.item())
        if (original, int(focus.item())) != tuple(action[:2]):
            raise RuntimeError("Original movement and focus did not match the frozen route logits.")
        telemetry = self.sessions[-1].setdefault("learned_contact", dict(
            eligible=0, changed=0, changed_direction_pairs=Counter(), event_examples=[], dropped_examples=0,
        ))
        telemetry["eligible"] += 1
        if move != original:
            telemetry["changed"] += 1
            telemetry["changed_direction_pairs"][f"{audit.MOVEMENTS[original]}->{audit.MOVEMENTS[move]}"] += 1
            examples = telemetry["event_examples"]
            if len(examples) < 256:
                examples.append(dict(stage_frame=env.last_frame, phase=self.last_live_phase,
                                     route_frame=self.actual_reference_frame,
                                     x=float(observation["context"][0]) * 192.,
                                     y=float(observation["context"][1]) * 448., focus=int(action[1]),
                                     base=original, candidate=move, base_score=float(scores[0, original]),
                                     candidate_score=float(scores[0, move])))
            else:
                telemetry["dropped_examples"] += 1
            action = action.copy()
            action[0] = move
        return action, state


def load_arms(args):
    dry, source = audit.load_route(args.artifact, args.demo_stage1, "phase-aware")
    risk_path = args.risk_model
    contact, _, _ = audit.load_contact(risk_path)
    shoot = load_arm(args.shoot_policy)
    if shoot.provenance["weights_sha256"] != SHOOT_FINGERPRINT or digest(args.bomb_model) != BOMB_FINGERPRINT:
        raise ValueError("The original phase actor's independent Shoot and learned Bomb sources must be unchanged.")
    bomb = json.loads(args.bomb_model.read_text(encoding="utf-8"))
    observed = args.development / "report.json"
    delay = args.development / "fatal-preemption-diagnostic.json"
    dry_run = json.loads(observed.read_text(encoding="utf-8"))
    sensitivity = json.loads(delay.read_text(encoding="utf-8"))
    if (dry_run["mode"] != "phase-aware" or dry_run["tape"]["sha256"] != sensitivity["archive_sha256"]
            or dry_run["analysis_source_sha256"] != digest(Path(audit.__file__))
            or dry_run["policy_source_sha256"] != digest(Path(phase.__file__))
            or dry_run["contact_head_sha256"] != digest(risk_path)
            or dry_run["route_model_sha256"] != source["model_sha256"]
            or dry_run["route_sha256"] != source["route_sha256"]):
        raise ValueError("The native trial needs the verified Phase Route and contact development lineage.")
    model, route = args.artifact / "model.pt", args.artifact / "route.npz"
    anchor = phase.PhaseRoute(model, route, bomb, shoot, "phase-aware")
    contender = ContactPhaseRoute(model, route, bomb, shoot, contact)
    anchor.knots = contender.knots = dry.knots
    provenance = dict(
        route_sha256=source["route_sha256"], route_model_sha256=source["model_sha256"], phase_knots=dry.knots,
        demo_replay=source["train_replay"], demo_stage1_sha256=digest(args.demo_stage1),
        independent_shoot=shoot.provenance, learned_bomb_value_sha256=BOMB_FINGERPRINT,
        phase_source_sha256=digest(Path(phase.__file__)),
    )
    candidate = dict(
        provenance, learned_contact_sha256=digest(risk_path), strength=PENALTY,
        development_audit_sha256=digest(observed), preemption_sensitivity_sha256=digest(delay),
        intervention="Frozen contact head subtracts risk from movement logits inside the original selected focus.",
        support="Active Stage 1, no held or requested X. Original focus, Shoot, and Bomb come from the phase actor.",
        training_prior="The offline contact head uses exogenous recorded fields, empirical motion and circular contact targets.",
        qualification="Observed-outcome effect remains untested. Coefficient chosen from offline audit and a post hoc two-action sensitivity case.",
    )
    return {"alice_phase": Arm(anchor, shoot.history, provenance),
            "alice_phase_learned_contact": Arm(contender, shoot.history, candidate)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=audit.DEFAULT_ROUTE)
    parser.add_argument("--demo-stage1", type=Path, default=audit.DEFAULT_DEMO)
    parser.add_argument("--risk-model", type=Path, default=audit.DEFAULT_RISK)
    parser.add_argument("--development", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--shoot-policy", type=Path, default=TEMPLATE)
    parser.add_argument("--bomb-model", type=Path, default=DEFAULT_BOMB)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--parakit", type=Path, default=audit.local.DATA.parents[1] / "work/parakit")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2601, 2602, 2603, 2604])
    parser.add_argument("--limit", type=int, default=3300)
    parser.add_argument("--focus-wait", type=int, default=90)
    parser.add_argument("--game-over-default-reference", type=Path)
    parser.add_argument("--game-over-retry-reference", type=Path)
    parser.add_argument("--menu-crop", action="append", nargs=4, type=int)
    parser.add_argument("--menu-max-error", type=float, default=.035)
    parser.add_argument("--transition-budget-mib", type=int, default=64)
    parser.add_argument("--transition-reserve-mib", type=int, default=20480)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.set_defaults(target="stage2", start_frame=30, crossover=[], timing_policy=[], restart_hook=None)
    args = parser.parse_args()
    refs = (args.game_over_default_reference, args.game_over_retry_reference, args.menu_crop)
    if (min(args.limit, args.transition_budget_mib) < 1 or args.focus_wait < 0 or args.transition_reserve_mib < 0
            or not 0 < args.menu_max_error < .2 or any(refs) and (not all(refs) or len(args.menu_crop) < 2)
            or len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2**63 or seed in (2001, 2201)
                                                               for seed in args.seeds)):
        parser.error("Use fresh distinct policy seeds, positive limits, and complete optional menu references.")
    if not args.verify_only and (args.out is None or os.name != "nt"):
        parser.error("Native evaluation requires Windows and --out.")
    torch.set_num_threads(1)
    try:
        arms = load_arms(args)
        spec = story.run_specification(args, arms)
        spec["controller"].update(kind="paired_phase_tracker_neural_contact", contact_strength=PENALTY,
                                  runtime_geometry=False, controls_compared=["movement"])
        spec["runtime"]["runner_sha256"] = digest(Path(__file__))
        sources = (story, phase, audit, train_demo_route, train_local_avoidance, fit_randomized_bomb_value,
                   train_native_bomb_timing)
        for mod in sources:
            spec["runtime"]["source_sha256"][mod.__name__] = digest(Path(mod.__file__))
        if args.verify_only:
            print(json.dumps(spec, indent=2), flush=True)
            return
        menu = GameOverMenu(*refs, args.menu_max_error) if all(refs) else None
        if not args.resume:
            args.out.mkdir(parents=True, exist_ok=False)
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    archive = native_story_transitions.configure(args, spec)
    telemetry = args.out / "controller-telemetry.json"
    with output_lock(args.out):
        sessions = json.loads(telemetry.read_text(encoding="utf-8")) if args.resume and telemetry.is_file() else []
        try:
            report = story.evaluate(args, arms, spec, args.out / "report.json", menu, None, transition_archive=archive)
        finally:
            sessions.append({name: arm.policy.sessions for name, arm in arms.items()})
            publish_json(telemetry, sessions)
    print(json.dumps(dict(status=report["status"], matched_summary=report.get("matched_summary"))), flush=True)


if __name__ == "__main__":
    main()
