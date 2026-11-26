"""Compare three frozen TH14 boss suffixes behind the same learned full-credit opening."""

import argparse
from collections import Counter
from functools import partial
import json
import os
from pathlib import Path

import torch

from scripts import checkpoint_storage, evaluate_bomb_crossover as crossover
from scripts import evaluate_full_credit as credit, evaluate_stage_policies, pretrain_policy
from scripts.checkpoint_storage import digest
from scripts.evaluate_sft_native import Arm, load_arm, load_ppo_arm, output_lock
from scripts.full_credit_reset import fresh_credit_reset
from scripts.train_credit_boss_ppo import comparable_observations
from scripts.train_full_policy import publish_json


START_FRAME = 30
HANDOFF_FRAME = 5000
HANDOFF_OVERSHOOT = 8
SUFFIX_SEED_BIT = 1 << 62
DEFAULT_INIT = crossover.DATA.parent / "th14-ppo397-bomb-distill-261125/policy.pt"
DEFAULT_PPO512 = crossover.DATA / "th14-stock-boss-ppo-f5000-512-261125/policy.zip"
DEFAULT_PPO4608 = crossover.DATA / "th14-stock-boss-ppo-f5000-4608-261125/policy.zip"
FINGERPRINTS = {
    "distillation_init": "77d87ee9bd9797dae9a142d2e6a5c231fc207dc054da4e5b02814d97eb6267b5",
    "ppo512": "c62710b5a01368e14953fc3a989e9903e371461e7f6f99ed30d989f80f677f75",
    "ppo4608": "cdedba02795b41ec2eaf7e8718ced6ccf7b54df03a3edfa8b883675ad2f9ea3c",
}


class FrozenSuffix:
    """Switch the policy once, at the first decision on or after the handoff frame."""

    def __init__(self, prefix, suffix):
        self.prefix, self.suffix = prefix, suffix
        self.environment = None
        self.handoff = None
        for field in ("motion_spans", "retain_recent_motion", "stage_clock"):
            setattr(self, field, getattr(suffix, field))

    def begin_credit(self, env, trial):
        self.environment = env
        seed = trial["policy_seed"]
        self.handoff = dict(requested_frame=HANDOFF_FRAME, prefix_seed=seed,
                            suffix_seed=seed ^ SUFFIX_SEED_BIT, reached=False)
        return self.handoff

    def predict(self, observation, deterministic=False):
        env = self.environment
        handoff = self.handoff
        if env is None or handoff is None:
            raise RuntimeError("Start and bind a verified fresh credit before sampling the comparator.")
        if handoff["reached"]:
            return self.suffix.predict(observation, deterministic=deterministic)
        if env.last_frame < HANDOFF_FRAME:
            return self.prefix.predict(observation, deterministic=deterministic)

        if env.last_frame > HANDOFF_FRAME + HANDOFF_OVERSHOOT:
            raise RuntimeError("Native timing overshot the permitted suffix handoff window.")
        native_stage, screen = env.current_stage(), env.current_screen()
        pause = env.game.read_game_val("game_state")
        continued = credit.continues(env)
        if (native_stage, env.snapshot.stage, screen, pause, continued) != (1, 1, 7, 2, 0):
            raise RuntimeError("The suffix handoff requires active Stage 1 on the original credit.")
        boss = env.snapshot.boss
        handoff.update(
            stage_frame=env.last_frame, native_stage_frame=env.stage_frame(),
            overshoot_frames=env.last_frame - HANDOFF_FRAME, native_stage=native_stage,
            native_pause_state=pause, native_continues=continued, player_state=env.snapshot.player_state,
            prefix_decisions=env.steps, suffix_first_decision=env.steps + 1,
            prefix_lives_lost=env.hits, prefix_bombs_consumed=env.bombs_used,
            resources=credit.resources(env), held_keys_before_release=sorted(env.held),
            boss=None if boss is None else dict(hp=boss.hp, max_hp=boss.max_hp,
                                               vulnerable=bool(boss.vulnerable), spell_id=env.snapshot.spell_id),
        )
        # The training rollout also releases the prefix's held inputs at this boundary.
        env.release()
        torch.manual_seed(handoff["suffix_seed"])
        action, state = self.suffix.predict(observation, deterministic=deterministic)
        handoff.update(reached=True, first_suffix_action=dict(zip(pretrain_policy.ACTION_NAMES, map(int, action))))
        return action, state


def load_comparator(args):
    prefix = crossover.crossover_arms(args)[crossover.SPLICE]
    template = load_arm(args.ppo_template)
    prefix_actor = Arm(prefix.policy.actor, prefix.history, prefix.provenance)
    candidates = {"distillation_init": load_arm(args.init_policy)}
    expected_stages = dict(title="TH14", version="1.00b", stage=1, difficulty="Normal", shot="Reimu A",
                           start_frame=START_FRAME, handoff_frame=HANDOFF_FRAME,
                           allowed_handoff_overshoot=HANDOFF_OVERSHOOT, stage2_only_success=True,
                           flags=dict(geometry=False, aim_correction=False, automatic_bomb=False,
                                      full_native_lives=True, continue_allowed=False,
                                      native_game_rng_controlled=False))
    training_configs = []
    parent_steps, parent_hash = 0, None
    for name, steps, path in (("ppo512", 512, args.ppo512), ("ppo4608", 4608, args.ppo4608)):
        arm = load_ppo_arm(path, template)
        record = json.loads(path.with_name("report.json").read_text(encoding="utf-8"))
        config = arm.provenance["training_configuration"]
        if (config["initialization_sha256"] != FINGERPRINTS["distillation_init"]
                or config["prefix"]["provenance"] != prefix.provenance or config["task"] != expected_stages
                or arm.provenance["native_training_decisions"] != steps or not arm.provenance["warm_start_actor_exact"]
                or record["specification"] != config or record["status"] != "complete"
                or record["starting_steps"] != parent_steps or record["resume_checkpoint_sha256"] != parent_hash
                or record["checkpoint_steps"] != steps or record["checkpoints"]["ppo_sha256"] != FINGERPRINTS[name]):
            raise ValueError(f"{name} must be the completed stock training archive from this exact suffix lineage.")
        arm.provenance["parent_archive_sha256"] = parent_hash
        arm.provenance["training_report_sha256"] = digest(path.with_name("report.json"))
        candidates[name] = arm
        training_configs.append(config)
        parent_steps, parent_hash = steps, FINGERPRINTS[name]
    if training_configs[0] != training_configs[1]:
        raise ValueError("The PPO4608 continuation must retain the original suffix training configuration.")
    if not all(arm.provenance["weights_sha256"] == FINGERPRINTS[name] for name, arm in candidates.items()):
        raise ValueError("The three comparator weights must match the frozen source fingerprints.")
    if not all(comparable_observations(arm, template) for arm in (prefix_actor, *candidates.values())):
        raise ValueError("Prefix and suffix policies must have identical native observation and action transforms.")
    return {
        name: Arm(FrozenSuffix(prefix.policy, arm.policy), arm.history,
                  dict(prefix=prefix.provenance, suffix=arm.provenance))
        for name, arm in candidates.items()
    }


def run_specification(args, arms):
    specification = credit.run_specification(args, arms)
    specification["controller"].update(
        kind="frozen_shared_prefix_suffix_comparison", handoff_frame=HANDOFF_FRAME,
        allowed_handoff_overshoot=HANDOFF_OVERSHOOT, release_prefix_keys_at_handoff=True,
        prefix_failure_counted=True, suffix_seed_xor=SUFFIX_SEED_BIT,
    )
    sources = (credit, crossover, evaluate_stage_policies, pretrain_policy, checkpoint_storage)
    specification["runtime"].update(
        runner_sha256=digest(Path(__file__)),
        source_sha256=dict(specification["runtime"]["source_sha256"],
                           **{mod.__name__: digest(Path(mod.__file__)) for mod in sources}),
    )
    return specification


def paired_results(report, names):
    """Retain all planned full-credit outcomes and compare within complete seed blocks."""
    groups = {}
    for item in report["completed"]:
        name, block = item["trial"]["arm"], item["trial"]["block"]
        result = item["result"]
        snapshot = groups.setdefault(block, dict(policy_seed=item["trial"]["policy_seed"], arms={}))
        snapshot["arms"][name] = dict(
            outcome=result["outcome"], stage2_clear=result["reached_goal"], stage2_entry=result["stage2_entry"],
            terminal_frame=result["native_terminal_stage_frame"], failure_frame=result["failure_frame"],
            deaths=result["deaths"], bombs_consumed=result["actual_bombs_consumed"],
            handoff=result["reset_info"]["suffix_handoff"],
        )
    matched = [row["arms"] for row in groups.values() if len(row["arms"]) == len(names)]
    comparisons = {}
    for challenger in names[1:]:
        wins = Counter((row[names[0]]["stage2_clear"], row[challenger]["stage2_clear"]) for row in matched)
        comparisons[challenger] = dict(reference=names[0], matched_blocks=len(matched),
                                       both_clear=wins[True, True], both_without_clear=wins[False, False],
                                       challenger_only_clear=wins[False, True], reference_only_clear=wins[True, False])
    return dict(status=report["status"], matched_blocks=len(matched),
                arms={name: dict(
                    handoffs=sum(row[name]["handoff"]["reached"] for row in matched),
                    before_handoff_outcomes=dict(Counter(row[name]["outcome"] for row in matched
                                                         if not row[name]["handoff"]["reached"])),
                ) for name in names}, stage2_clear_pairs=comparisons, by_block=groups)


def run_comparison(args, arms, specification):
    def reset(env, trial):
        observation, info = fresh_credit_reset(env, trial, start_frame=args.start_frame)
        handoff = arms[trial["arm"]].policy.begin_credit(env, trial)
        return observation, dict(info, suffix_handoff=handoff)

    path = args.out / "report.json"
    try:
        report = credit.run_evaluation(args, arms, path, specification, reset_callback=reset)
    finally:
        for arm in arms.values():
            arm.policy.environment = None
        if path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("specification") == specification:
                publish_json(args.out / "paired-results.json", paired_results(saved, list(arms)))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-policy", type=Path, default=DEFAULT_INIT)
    parser.add_argument("--ppo512", type=Path, default=DEFAULT_PPO512)
    parser.add_argument("--ppo4608", type=Path, default=DEFAULT_PPO4608)
    parser.add_argument("--ppo211", type=Path, default=crossover.PPO211)
    parser.add_argument("--ppo397", type=Path, default=crossover.PPO397)
    parser.add_argument("--ppo-template", type=Path, default=crossover.TEMPLATE)
    parser.add_argument("--parakit", type=Path, default=crossover.REPO.parent / "parakit")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(2401, 2405)))
    parser.add_argument("--limit", type=int, default=3300, help="Decision cap for a complete Stage 1 credit.")
    parser.add_argument("--focus-wait", type=int, default=90)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true", help="Validate frozen assets and show the schedule offline.")
    parser.set_defaults(start_frame=START_FRAME)
    args = parser.parse_args()
    if not args.verify_only and (args.out is None or os.name != "nt"):
        parser.error("Native evaluation requires Windows and --out.")
    if args.limit <= (HANDOFF_FRAME - START_FRAME) // 4 + 1 or args.focus_wait < 0:
        parser.error("Use a credit decision cap that reaches the suffix and a nonnegative focus wait.")
    if len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2**63 for seed in args.seeds):
        parser.error("Use distinct policy seeds in [0, 2**63).")
    torch.set_num_threads(1)
    try:
        arms = load_comparator(args)
        specification = run_specification(args, arms)
        if args.verify_only:
            print(json.dumps(specification, indent=2), flush=True)
            return
        if not args.resume:
            args.out.mkdir(parents=True, exist_ok=False)
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    with output_lock(args.out):
        report = run_comparison(args, arms, specification)
    print(json.dumps(dict(matched_blocks=report["matched_blocks"], summary=report["matched_summary"])), flush=True)


if __name__ == "__main__":
    main()
