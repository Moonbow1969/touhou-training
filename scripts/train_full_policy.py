"""Warm-start full-policy PPO and compare controllers in the native TH14 executable."""

import argparse
from functools import partial
from hashlib import file_digest
import json
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3 import __version__ as sb3_version
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import torch

from scripts.evaluate_boss_aim import AimHead, trial
from scripts.evaluate_stage_policies import policy_observation, read_policy
from scripts.pretrain_policy import ACTION_NAMES, ACTION_WIDTHS, MovementPolicy, network_kwargs
from scripts.stage_policy import NormalStageOne


UPDATE_STEPS = 256
CONTROLLERS = ("sft", "ppo", "hybrid")
PPO_SETTINGS = dict(n_steps=UPDATE_STEPS, batch_size=64, n_epochs=4, learning_rate=8e-5,
                    gamma=.997, gae_lambda=.95, clip_range=.15, ent_coef=.004)
REWARDS = {
    "sparse": {"clear_or_survive_horizon": 1, "miss_or_native_end": -1, "other": 0},
    "survival-time": {"alive_step": "1 / train_limit", "miss_or_native_end": 0, "stage_clear": 1},
}


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def publish_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class SurvivalRollout(gym.Wrapper):
    def __init__(self, env, base, reward_mode):
        super().__init__(env)
        self.base = base
        self.reward_mode = reward_mode
        self.observation_space = base.observation_space
        self.step_seconds = 0.
        self.reset_seconds = 0.
        self.resets = 0
        self.action_counts = {name: np.zeros(width, dtype=np.int64)
                              for name, width in zip(ACTION_NAMES, ACTION_WIDTHS)}
        self.frame_deltas = {}

    def reset(self, **kwargs):
        started = perf_counter()
        observation, info = self.env.reset(**kwargs)
        observation = policy_observation(self.base, self.env, observation)
        self.reset_seconds += perf_counter() - started
        self.resets += 1
        return observation, info

    def step(self, action):
        started = perf_counter()
        frame = self.env.last_frame
        observation, _, terminated, truncated, info = self.env.step(action)
        observation = policy_observation(self.base, self.env, observation)
        for counts, value in zip(self.action_counts.values(), action):
            counts[value] += 1
        delta = self.env.last_frame - frame
        self.frame_deltas[delta] = self.frame_deltas.get(delta, 0) + 1
        successful = truncated or info["outcome"] == "advanced_stage"
        # Reaching the fixed survival horizon is a terminal outcome of this task.
        # SB3 must not bootstrap its value through that boundary.
        if self.reward_mode == "sparse":
            reward = 1. if successful else -1. if terminated else 0.
        elif info["outcome"] == "advanced_stage":
            reward = 1.
        else:
            reward = 0. if terminated else 1. / self.env.limit
        self.step_seconds += perf_counter() - started
        return observation, reward, terminated or truncated, False, info


class RolloutLog(BaseCallback):
    def __init__(self, report, out, env):
        super().__init__()
        self.report, self.out, self.env = report, out, env
        self.started = perf_counter()
        self.rollout_started = self.started
        self.rollout_finished = None
        self.step_start = 0.
        self.reset_start = 0.

    def _on_step(self):
        return True

    def _on_rollout_start(self):
        self.save_update()
        self.rollout_started = perf_counter()
        self.step_start = self.env.step_seconds
        self.reset_start = self.env.reset_seconds

    def _on_rollout_end(self):
        self.rollout_finished = perf_counter()

    def _on_training_end(self):
        self.save_update()
        self.save_checkpoint()

    def save_checkpoint(self):
        if self.report.get("checkpoint_steps") == self.num_timesteps:
            return
        checkpoint = self.out / "policy.zip"
        temporary = self.out / "policy.tmp.zip"
        self.model.save(temporary)
        temporary.replace(checkpoint)
        self.report["checkpoints"]["ppo_sha256"] = digest(checkpoint)
        self.report["checkpoint_steps"] = self.num_timesteps
        publish_json(self.out / "report.json", self.report)

    def save_update(self):
        if self.rollout_finished is None:
            return
        elapsed = perf_counter() - self.rollout_finished
        row = dict(steps=self.num_timesteps, optimization_epochs=self.model._n_updates,
                   rollout_seconds=self.rollout_finished - self.rollout_started,
                   environment_step_seconds=self.env.step_seconds - self.step_start,
                   reset_seconds=self.env.reset_seconds - self.reset_start,
                   update_and_logging_seconds=elapsed,
                   completed_episodes=len(self.env.env.completed))
        row.update({name.removeprefix("train/"): float(value)
                    for name, value in self.logger.name_to_value.items()
                    if name.startswith("train/") and np.isfinite(value)})
        self.report["updates"].append(row)
        self.report["training_episodes"] = self.env.env.completed.copy()
        self.report["training_timing"] = dict(seconds=perf_counter() - self.started,
                                              native_step_seconds=self.env.step_seconds,
                                              reset_seconds=self.env.reset_seconds,
                                              resets=self.env.resets,
                                              native_frame_deltas=self.env.frame_deltas.copy())
        self.report["training_action_counts"] = {name: counts.tolist()
                                                  for name, counts in self.env.action_counts.items()}
        publish_json(self.out / "report.json", self.report)
        print(json.dumps(row), flush=True)
        if self.num_timesteps % 1024 == 0:
            self.save_checkpoint()
        self.rollout_finished = None


def warm_start(base, env, seed):
    width = base.mlp_extractor.policy_net[0].out_features
    encoder = "attention" if hasattr(base.features_extractor, "keys") else "pool"
    model = PPO(MovementPolicy, env, policy_kwargs=network_kwargs(width, encoder),
                device="cpu", seed=seed, **PPO_SETTINGS)
    model.policy.load_state_dict(base.state_dict(), strict=True)
    # Behaviour cloning trains the shared encoder and actor, while PPO needs a fresh value branch.
    model.policy.mlp_extractor.value_net.apply(partial(model.policy.init_weights, gain=np.sqrt(2)))
    model.policy.value_net.apply(partial(model.policy.init_weights, gain=1.))
    observation = base.observation_space.sample()
    with torch.no_grad():
        before = base.get_distribution(base.obs_to_tensor(observation)[0]).distribution
        after = model.policy.get_distribution(model.policy.obs_to_tensor(observation)[0]).distribution
        for sft, ppo in zip(before, after):
            torch.testing.assert_close(sft.logits, ppo.logits, rtol=0, atol=0)
    return model


def direct_trial(stage, base, policy):
    observation, _ = stage.reset()
    start_frame = stage.last_frame
    action_counts = {name: np.zeros(width, dtype=np.int64) for name, width in zip(ACTION_NAMES, ACTION_WIDTHS)}
    while True:
        action, _ = policy.predict(policy_observation(base, stage, observation), deterministic=True)
        for counts, value in zip(action_counts.values(), action):
            counts[value] += 1
        observation, _, terminated, truncated, info = stage.step(action)
        if terminated or truncated:
            return dict(info, start_frame=start_frame,
                        action_counts={name: counts.tolist() for name, counts in action_counts.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--aim", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parakit", type=Path, default=Path(__file__).resolve().parents[2] / "parakit")
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--train-limit", type=int, default=512)
    parser.add_argument("--eval-limit", type=int, default=3300)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--reward", choices=tuple(REWARDS), default="sparse")
    parser.add_argument("--resume", type=Path, help="Resume a PPO checkpoint from the same SFT parent.")
    parser.add_argument("--focus-wait", type=int, default=30)
    args = parser.parse_args()
    if args.steps < 0 or args.steps % UPDATE_STEPS or not (0 < args.train_limit <= args.eval_limit):
        parser.error(f"Steps must be a multiple of {UPDATE_STEPS}, and 0 < train limit <= eval limit.")
    if args.blocks < 0 or not (args.blocks or args.steps) or (not args.steps and not args.resume):
        parser.error("Request training, evaluation of a saved checkpoint, or both.")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    base = read_policy(args.base, "cpu", motion_history=True)
    if base.factorized_movement:
        parser.error("This comparison requires a categorical SFT movement head.")
    student = AimHead()
    student.load_state_dict(torch.load(args.aim, map_location="cpu", weights_only=True))
    student.eval()
    report = {
        "task": {"title": "TH14", "version": "1.00b", "difficulty": "Normal", "shot": "Reimu A",
                 "first_stage": 1, "last_stage": 1, "first_miss_ends_trial": True, "frames_per_action": 4,
                 "train_limit": args.train_limit, "eval_limit": args.eval_limit},
        "checkpoints": {"base_sha256": digest(args.base), "aim_sha256": digest(args.aim)},
        "runtime": {"runner_sha256": digest(Path(__file__)),
                    "parakit_offsets_sha256": digest(args.parakit / "offsets.py"),
                    "torch": str(torch.__version__), "stable_baselines3": sb3_version, "device": "cpu"},
        "seed": args.seed, "additional_steps": args.steps, "requested_blocks": args.blocks,
        "ppo_hyperparameters": PPO_SETTINGS,
        "reward": REWARDS[args.reward],
        "controller_settings": {arm: {"geometry": arm == "hybrid", "aim_correction": arm == "hybrid",
                                      "automatic_bomb": arm == "hybrid"} for arm in CONTROLLERS},
        "controllers": {arm: [] for arm in CONTROLLERS}, "updates": [], "status": "started",
    }
    if args.resume:
        prior = json.loads(args.resume.with_name("report.json").read_text(encoding="utf-8"))
        if (prior["checkpoints"]["base_sha256"] != report["checkpoints"]["base_sha256"]
                or prior["task"]["train_limit"] != args.train_limit or prior["seed"] != args.seed
                or prior["ppo_hyperparameters"] != PPO_SETTINGS
                or prior["reward"] != REWARDS[args.reward]
                or digest(args.resume) != prior["checkpoints"].get("ppo_sha256")):
            parser.error("The resume checkpoint, SFT parent, training task, or settings do not match.")
        report["checkpoints"]["resume_sha256"] = prior["checkpoints"]["ppo_sha256"]
    report_path = args.out / "report.json"
    publish_json(report_path, report)
    stage = None
    try:
        stage = NormalStageOne(args.parakit, limit=args.train_limit, motion_history=True,
                               history_frames=max(base.motion_spans or (12,)), focus_wait=args.focus_wait)
        env = SurvivalRollout(stage, base, args.reward)
        model = PPO.load(args.resume, env=env, device="cpu") if args.resume else warm_start(base, env, args.seed)
        report["starting_steps"] = model.num_timesteps
        report["warm_start_actor_exact"] = not args.resume or prior["warm_start_actor_exact"]
        if args.steps:
            model.set_logger(configure(str(args.out), ["csv"]))
            model.learn(args.steps, callback=RolloutLog(report, args.out, env), reset_num_timesteps=not args.resume)
        report["trained_steps"] = model.num_timesteps
        report["actor_head_weight_delta_l2"] = {
            name: float(torch.linalg.vector_norm(delta))
            for name, delta in zip(ACTION_NAMES,
                                   (model.policy.action_net.weight.detach() - base.action_net.weight.detach())
                                   .split(ACTION_WIDTHS))}
        report["status"] = "evaluating"
        publish_json(report_path, report)
        stage.limit = args.eval_limit
        for block in range(args.blocks):
            order = CONTROLLERS[block % len(CONTROLLERS):] + CONTROLLERS[:block % len(CONTROLLERS)]
            for name in order:
                start = perf_counter()
                print(f"Evaluating block {block + 1}, {name}.", flush=True)
                if name == "hybrid":
                    result = trial(stage, base, None, True, lambda *_: None, None, student, True, True, True)
                else:
                    result = direct_trial(stage, base, base if name == "sft" else model.policy)
                result.update(block=block + 1, seconds=perf_counter() - start)
                report["controllers"][name].append(result)
                publish_json(report_path, report)
                keys = ("outcome", "stage_frame", "first_miss_frame", "score", "misses", "bombs_used", "seconds")
                print(name, json.dumps({key: result[key] for key in keys}), flush=True)
        report["status"] = "complete"
    except BaseException as error:
        report["status"] = "interrupted"
        report["error_type"] = type(error).__name__
        raise
    finally:
        if stage is not None:
            stage.close()
        publish_json(report_path, report)


if __name__ == "__main__":
    main()
