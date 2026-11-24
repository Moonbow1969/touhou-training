"""Compare frozen SFT and PPO checkpoints in the original TH14 1.00b Stage 1."""

import argparse
import ctypes
import io
import json
import os
import re
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3 import __version__ as sb3_version

from scripts import evaluate_stage_policies, stage_policy, train_full_policy
from scripts.checkpoint_storage import checkpoint_manifest, digest, read_resume
from scripts.evaluate_stage_policies import summarize
from scripts.pretrain_policy import (
    MOTION_INTERVALS,
    MovementPolicy,
    action_space,
    network_kwargs,
    observation_space,
    predict_heads,
)
from scripts.stage_policy import NormalStageOne
from scripts.train_full_policy import direct_trial, publish_json, retry_story_game_over


@dataclass
class Arm:
    policy: MovementPolicy
    history: bool
    provenance: dict


class HeadSampler:
    """Sample movement independently of the focus, shoot and Bomb decode."""

    def __init__(self, policy, seed, greedy_buttons):
        self.policy = policy
        self.greedy_buttons = greedy_buttons
        self.movement_rng = torch.Generator().manual_seed(seed)
        self.button_rng = torch.Generator().manual_seed(seed ^ (1 << 62))

    @torch.inference_mode()
    def predict(self, observation, deterministic=False):
        tensors, _ = self.policy.obs_to_tensor(observation)
        movement, *buttons = predict_heads(self.policy, tensors["bullets"], tensors["context"])
        direction = torch.multinomial(movement.softmax(dim=-1), 1, generator=self.movement_rng).item()
        if self.greedy_buttons:
            bits = [head.argmax(dim=-1).item() for head in buttons]
        else:
            bits = [torch.multinomial(head.softmax(dim=-1), 1, generator=self.button_rng).item() for head in buttons]
        return np.asarray([direction, *bits], dtype=np.int64), None


def load_arm(path):
    payload = path.read_bytes()
    weights = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    metadata = path.with_name("metrics.json")
    sidecar = metadata.read_bytes() if metadata.is_file() else None
    report = json.loads(sidecar) if sidecar is not None else {}
    options = dict(report.get("config", {}), **report)
    hidden, input_features = weights["mlp_extractor.policy_net.0.weight"].shape
    encoder = "attention" if "features_extractor.keys.weight" in weights else "pool"
    # Both SFT encoders append the policy context to 192 pooled projectile features.
    matching = [(history, retained, clock)
                for history, retained in ((False, False), (True, False), (True, True))
                for clock in (False, True)
                if 192 + observation_space(history, retained, clock)["context"].shape[0] == input_features]
    if len(matching) != 1:
        raise ValueError(f"{path.name} has an unsupported SFT context width.")
    history, retained, clock = matching[0]
    expected = dict(hidden=hidden, encoder=encoder, motion_history=history,
                    retain_recent_motion=retained, stage_clock=clock)
    for field, value in expected.items():
        if field in options and (options[field] != value):
            raise ValueError(f"{path.name} has inconsistent SFT metadata for {field}.")
    spans = options.get("motion_spans")
    if spans is not None and (not history or len(spans) != MOTION_INTERVALS
                              or any(type(n) is not int or n < 4 or n > 120 for n in spans)
                              or list(spans) != sorted(set(spans))):
        raise ValueError(f"{path.name} has unsupported motion-history spans.")
    if retained and spans is None:
        raise ValueError(f"{path.name} requires its metrics.json to identify the extended motion history.")
    policy = MovementPolicy(observation_space(history, retained, clock), action_space(), lambda _: 0.,
                            **network_kwargs(hidden, encoder))
    policy.load_state_dict(weights, strict=True)
    policy.motion_spans = spans
    policy.retain_recent_motion = retained
    policy.stage_clock = clock
    policy.factorized_movement = bool(options.get("factorized_movement", False))
    policy.eval()
    evidence = dict(**expected, motion_spans=spans, factorized_movement=policy.factorized_movement,
                    weights_sha256=sha256(payload).hexdigest(),
                    metadata_sha256=sha256(sidecar).hexdigest() if sidecar is not None else None)
    return Arm(policy, history, evidence)


def load_ppo_arm(path, template):
    manifest = checkpoint_manifest(path)
    if manifest is None or manifest["compatibility"].get("base_sha256") != template.provenance["weights_sha256"]:
        raise ValueError(f"{path.name} needs a checkpoint manifest matching the PPO observation template.")
    fingerprint, warm_exact, evidence = read_resume(path, manifest["compatibility"])
    archive = PPO.load(path, device="cpu").policy
    policy = deepcopy(template.policy)
    if archive.observation_space != policy.observation_space or archive.action_space != policy.action_space:
        raise ValueError(f"{path.name} has different observation or action spaces from its template.")
    policy.load_state_dict(archive.state_dict(), strict=True)
    policy.eval()
    provenance = dict(template.provenance, weights_sha256=fingerprint, archive_kind="ppo",
                      template_sha256=template.provenance["weights_sha256"], native_training_decisions=manifest["steps"],
                      warm_start_actor_exact=warm_exact, lineage_evidence=evidence,
                      training_configuration=manifest["compatibility"])
    return Arm(policy, template.history, provenance)


def load_arms(selections, ppo_selections=(), ppo_template=None):
    arms = {}
    template = load_arm(ppo_template) if ppo_selections else None
    for group, loader in ((selections, load_arm),
                          (ppo_selections, lambda path: load_ppo_arm(path, template))):
        for specification in group:
            name, divider, path = specification.partition("=")
            if not divider or not path or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", name) or name in arms:
                raise ValueError("Use distinct NAME=CHECKPOINT arguments across all arms.")
            arms[name] = loader(Path(path))
    return arms


@contextmanager
def output_lock(directory):
    with (directory / ".evaluation-lock").open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def trial_schedule(arms, seeds):
    names = list(arms)
    trials = []
    for index, seed in enumerate(seeds):
        pivot = index % len(names)
        for position, name in enumerate(names[pivot:] + names[:pivot], 1):
            trials.append(dict(block=index + 1, order=position, arm=name, policy_seed=seed))
    return trials


def run_specification(args, arms):
    sources = (evaluate_stage_policies, stage_policy, train_full_policy)
    return {
        "schema": 1,
        "task": {
            "title": "TH14", "version": "1.00b", "difficulty": "Normal", "shot": "Reimu A",
            "stage": 1, "decision_limit": args.limit, "max_misses": args.max_misses, "frames_per_action": 4,
        },
        "controller": {
            "kind": "frozen_sft_and_ppo" if any(arm.provenance.get("archive_kind") for arm in arms.values()) else "sft",
            "decode": "split_heads" if args.greedy_buttons else "sampled",
            "geometry": False, "aim_correction": False,
            "automatic_bomb": False, "native_rng_seed_controlled": False,
            "button_crossover": {
                "movement_rng": "independent torch.Generator seeded with the policy seed",
                "sampled_button_rng": "independent torch.Generator with the policy seed XOR 2^62",
                "greedy_button_arms": sorted(args.greedy_buttons),
            } if args.greedy_buttons else None,
        },
        "runtime": {
            "runner_sha256": digest(Path(__file__)),
            "source_sha256": {module.__name__: digest(Path(module.__file__)) for module in sources},
            "parakit_offsets_sha256": digest(args.parakit / "offsets.py"),
            "torch": str(torch.__version__), "stable_baselines3": sb3_version, "device": "cpu",
        },
        "policies": {name: arm.provenance for name, arm in arms.items()},
        "schedule": trial_schedule(arms, args.seeds),
    }


def restore_report(path, specification):
    if not path.exists():
        raise FileNotFoundError("No existing evaluation report to resume.")
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("specification") != specification:
        raise ValueError("Resume requires the same checkpoints, task, runner and rotation schedule.")
    schedule = specification["schedule"]
    completed = saved["completed"]
    if len(completed) > len(schedule) or [row["trial"] for row in completed] != schedule[:len(completed)]:
        raise ValueError("Saved episodes are not a verified prefix of the planned schedule.")
    pending = saved["pending"]
    if pending is not None:
        if len(completed) == len(schedule) or pending != schedule[len(completed)]:
            raise ValueError("Saved pending trial conflicts with the planned schedule.")
        saved["restarts"].append(pending)
        saved["pending"] = None
    return saved


def summarize_arm(episodes):
    result = summarize(episodes)
    inputs = sum(episode["horizontal_commands_while_active"] for episode in episodes)
    responses = sum(episode["horizontal_responses"] for episode in episodes)
    counts = [sum(episode["action_counts"]["movement"][index] for episode in episodes) for index in range(9)]
    return dict(result, outcomes=dict(Counter(episode["outcome"] for episode in episodes)),
                horizontal_commands_while_active=inputs, horizontal_responses=responses,
                horizontal_response_fraction=responses / inputs if inputs else None,
                left_actions=counts[0] + counts[3] + counts[6], right_actions=counts[2] + counts[5] + counts[8],
                idle_actions=counts[4],
                median_x_range=median(episode["x_max"] - episode["x_min"] for episode in episodes) if episodes else None)


def refresh_summary(report, arms):
    count = len(report["completed"]) // len(arms)
    matched = report["completed"][:count * len(arms)]
    report["matched_blocks"] = count
    report["matched_summary"] = {
        name: summarize_arm([row["result"] for row in matched if row["trial"]["arm"] == name]) for name in arms
    }


@contextmanager
def native_stage(args, arms):
    keep_awake = ctypes.windll.kernel32.SetThreadExecutionState
    continuous = 0x80000000
    keep_awake(continuous | 0x00000001 | 0x00000002)
    stage = None
    try:
        if args.retry_game_over:
            retry_story_game_over(args.parakit)
        longest_history = max([12] + [n for arm in arms.values() for n in (arm.policy.motion_spans or [])])
        stage = NormalStageOne(args.parakit, limit=args.limit, max_misses=args.max_misses,
                               motion_history=any(arm.history for arm in arms.values()),
                               history_frames=longest_history, focus_wait=args.focus_wait)
        yield stage
    finally:
        try:
            if stage is not None:
                stage.close()
        finally:
            keep_awake(continuous)


def run_evaluation(args, arms, path, specification, stage_session=native_stage):
    if args.resume:
        report = restore_report(path, specification)
    else:
        report = dict(specification=specification, status="started", pending=None, completed=[], restarts=[])
    refresh_summary(report, arms)
    remaining = specification["schedule"][len(report["completed"]):]
    if not remaining:
        report.update(status="complete", error=None)
        publish_json(path, report)
        return report
    report.update(status="evaluating", error=None)
    publish_json(path, report)
    try:
        with stage_session(args, arms) as stage:
            for trial in remaining:
                report["pending"] = trial
                publish_json(path, report)
                print(json.dumps(dict(status="trial", **trial)), flush=True)
                arm = arms[trial["arm"]]
                stage.motion_history = arm.history
                if args.greedy_buttons:
                    actor = HeadSampler(arm.policy, trial["policy_seed"], trial["arm"] in args.greedy_buttons)
                else:
                    torch.manual_seed(trial["policy_seed"])
                    actor = arm.policy
                start = perf_counter()
                result = direct_trial(stage, arm.policy, actor, deterministic=False)
                result["seconds"] = perf_counter() - start
                report["completed"].append(dict(trial=trial, result=result))
                report["pending"] = None
                refresh_summary(report, arms)
                publish_json(path, report)
                fields = ("outcome", "stage_frame", "first_miss_frame", "horizontal_commands_while_active",
                          "horizontal_responses")
                print(json.dumps(dict(status="completed", **trial, **{key: result[key] for key in fields})), flush=True)
    except BaseException as error:
        report.update(status="interrupted", error=dict(type=type(error).__name__, detail=str(error)))
        publish_json(path, report)
        raise
    report["status"] = "complete"
    publish_json(path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", action="append", default=[], metavar="NAME=CHECKPOINT")
    parser.add_argument("--ppo-policy", action="append", default=[], metavar="NAME=ARCHIVE")
    parser.add_argument("--ppo-template", type=Path, help="SFT checkpoint giving the PPO observation transformation.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(61, 65)), metavar="SEED")
    parser.add_argument("--out", type=Path, required=True, help="Directory for the atomically updated report.json.")
    parser.add_argument("--parakit", type=Path, default=Path(__file__).resolve().parents[2] / "parakit")
    parser.add_argument("--limit", type=int, default=3300)
    parser.add_argument("--max-misses", type=int, choices=(1, 2), default=1)
    parser.add_argument("--focus-wait", type=int, default=30)
    parser.add_argument("--retry-game-over", action="store_true",
                        help="Recover a verified Normal Reimu A Stage 1 Retry menu before entering.")
    parser.add_argument("--greedy-buttons", nargs="+", default=[], metavar="ARM",
                        help="Use argmax buttons in these arms and a separate movement RNG in every arm.")
    parser.add_argument("--resume", action="store_true", help="Resume the same run after its last committed episode.")
    args = parser.parse_args()
    if not (args.policy or args.ppo_policy) or bool(args.ppo_policy) != bool(args.ppo_template):
        parser.error("Specify at least one arm and provide --ppo-template with --ppo-policy.")
    if args.limit < 1 or args.focus_wait < 0 or len(args.seeds) != len(set(args.seeds)):
        parser.error("Use a positive decision limit, nonnegative focus wait, and distinct policy seeds.")
    if any(not 0 <= seed < 2 ** 63 for seed in args.seeds):
        parser.error("Use policy seeds in [0, 2**63).")
    if os.name != "nt":
        parser.error("The native TH14 1.00b game adapter requires Windows.")
    torch.set_num_threads(1)
    try:
        arms = load_arms(args.policy, args.ppo_policy, args.ppo_template)
        if args.greedy_buttons and not set(args.greedy_buttons) < set(arms):
            raise ValueError("Name a subset of arms to use greedy buttons and leave a sampled control.")
        specification = run_specification(args, arms)
        if args.resume:
            if not args.out.is_dir():
                raise FileNotFoundError("No evaluation output directory to resume.")
        else:
            args.out.mkdir(parents=True, exist_ok=False)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    with output_lock(args.out):
        report = run_evaluation(args, arms, args.out / "report.json", specification)
    print(json.dumps(dict(matched_blocks=report["matched_blocks"], summary=report["matched_summary"])), flush=True)


if __name__ == "__main__":
    main()
