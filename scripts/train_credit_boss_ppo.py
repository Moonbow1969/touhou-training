"""Train a single TH14 PPO actor after a frozen learned Stage 1 prefix."""

import argparse
from collections import Counter
import json
from math import isfinite
import os
from pathlib import Path
from time import perf_counter

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import torch

from scripts import evaluate_bomb_crossover, evaluate_full_credit, full_credit_reset, stage_policy, train_full_policy
from scripts.audit_th14_clear_objective import NativeState, classify
from scripts.checkpoint_storage import CheckpointStore, MIB, checkpoint_manifest, digest, read_resume
from scripts.evaluate_bomb_crossover import PPO211, PPO397, SPLICE, TEMPLATE, crossover_arms
from scripts.evaluate_boss_aim import same_phase
from scripts.evaluate_full_credit import continues, native_stage, resources, verify_opening
from scripts.evaluate_sft_native import load_arm, load_ppo_arm, output_lock
from scripts.evaluate_stage_policies import policy_observation
from scripts.train_full_policy import PPO_SETTINGS as SURVIVAL_SETTINGS, publish_json, warm_start

# A finite credit gives the same return for any genuine Stage 2 entry.
PPO_SETTINGS = dict(SURVIVAL_SETTINGS, gamma=1.)


def action_observation(arm, stage, observation):
    return policy_observation(arm.policy, stage, observation, uses_history=arm.history)


def verified_damage(before, after, native_stage_number, native_state):
    boss = before.boss
    if (native_stage_number != 1 or native_state != 2 or boss is None or not boss.vulnerable
            or not same_phase(before, after) or not after.boss.vulnerable):
        return 0
    damage = boss.hp - after.boss.hp
    return damage if 0 < damage <= 512 else 0


def local_reward(outcome, before, after, gamma):
    terminal = outcome in ("clear", "game_over")
    successor = 0. if terminal else after
    return float(outcome == "clear") + gamma * successor - before


def native_outcome(stage, info, capped):
    native = NativeState(info["native_stage"], stage.current_screen(), info["native_pause_state"],
                         stage.game.read_game_val("lives"), continues(stage))
    return classify("stage", 1, native, training_limit_reached=capped)


class BossRollout(gym.Env):
    """Run the prefix during reset, outside the PPO rollout buffer."""

    def __init__(self, stage, prefix, student, args, log, prefix_cursor=0):
        super().__init__()
        self.stage, self.prefix, self.student = stage, prefix, student
        self.args, self.log = args, log
        self.observation_space = student.policy.observation_space
        self.action_space = student.policy.action_space
        self.prefix_cursor = prefix_cursor
        self.accepted_steps = 0
        self.step_seconds = 0.
        self.reset_seconds = 0.
        self.stage.motion_history = prefix.history or student.history
        self.current = None
        self.frame_deltas = Counter()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        started = perf_counter()
        for _ in self.args.prefix_seeds:
            index = self.prefix_cursor
            policy_seed = self.args.prefix_seeds[index % len(self.args.prefix_seeds)]
            self.prefix_cursor += 1
            observation, reset = full_credit_reset.fresh_credit_reset(self.stage, start_frame=self.args.start_frame)
            verify_opening(self.stage, self.args.start_frame)
            begin = perf_counter()
            status = self.advance_prefix(observation, policy_seed)
            self.stage.release()
            self.log["prefix_attempt_cursor"] = self.prefix_cursor
            row = dict(status, policy_seed=policy_seed, prefix_attempt=index + 1,
                       reset=reset["fresh_credit_reset"], seconds=perf_counter() - begin,
                       entry_resources=resources(self.stage))
            self.log["prefix_attempts"].append(row)
            self.log["current"] = None
            if status["outcome"] == "handoff":
                self.reset_seconds += perf_counter() - started
                self.current = dict(handoff=row, first_suffix_miss_frame=None, section_decisions=0,
                                    section_deaths=0, section_bombs_consumed=0, reward=0.,
                                    verified_hp_damage=0, verified_damage_fraction=0.,
                                    before_lives_lost=self.stage.hits, before_bombs_used=self.stage.bombs_used)
                self.log["current"] = self.current
                self.save_report()
                return action_observation(self.student, self.stage, self.stage.observation()), {}
            self.save_report()
        raise RuntimeError("All scheduled learned-prefix seeds failed to reach the section; no invalid prefix was trained.")

    def advance_prefix(self, observation, policy_seed):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(policy_seed)
            while self.stage.last_frame < self.args.handoff_frame:
                action, _ = self.prefix.policy.predict(
                    action_observation(self.prefix, self.stage, observation), deterministic=False,
                )
                observation, _, ended, truncated, info = self.stage.step(action)
                if continues(self.stage):
                    raise RuntimeError("The frozen prefix entered a continued credit.")
                if ended or truncated:
                    outcome = native_outcome(self.stage, info, truncated)
                    if outcome == "ongoing":
                        raise RuntimeError("The prefix adapter stopped on an active credit.")
                    return dict(outcome=outcome, stage_frame=self.stage.last_frame,
                                stage_decisions=self.stage.steps, deaths=self.stage.hits,
                                bombs_consumed=self.stage.bombs_used)
        if self.stage.last_frame > self.args.handoff_frame + 8:
            raise RuntimeError("Native timing overshot the permitted suffix handoff window.")
        return dict(outcome="handoff", stage_frame=self.stage.last_frame, stage_decisions=self.stage.steps,
                    deaths=self.stage.hits, bombs_consumed=self.stage.bombs_used,
                    player_state=self.stage.snapshot.player_state)

    def save_report(self):
        self.log["timing"] = dict(student_step_seconds=self.step_seconds, reset_and_prefix_seconds=self.reset_seconds,
                                  student_native_frame_deltas=self.frame_deltas)
        publish_json(self.args.out / "report.json", self.log)

    def step(self, action):
        started = perf_counter()
        row = self.current
        before = self.stage.snapshot
        old_frame = self.stage.last_frame
        observation, _, native_terminated, native_truncated, info = self.stage.step(action)
        if continues(self.stage):
            self.stage.release()
            raise RuntimeError("A Continue invalidated the current native PPO credit.")
        row["section_decisions"] += 1
        self.accepted_steps += 1
        tick_delta = self.stage.last_frame - old_frame
        if info["native_stage"] == 1 and info["native_pause_state"] == 2:
            self.frame_deltas[tick_delta] += 1
        capped = native_truncated or row["section_decisions"] >= self.args.section_limit
        outcome = native_outcome(self.stage, info, capped) if native_terminated or capped else "ongoing"
        if native_terminated and outcome == "ongoing":
            raise RuntimeError("The native adapter stopped on an active credit.")
        terminated = outcome in ("clear", "game_over")
        truncated = outcome == "time_limit"
        prior_potential = self.args.damage_potential * min(1., row["verified_damage_fraction"])
        damage = verified_damage(before, self.stage.snapshot, info["native_stage"], info["native_pause_state"])
        row["verified_hp_damage"] += damage
        if damage:
            row["verified_damage_fraction"] += damage / before.boss.max_hp
        potential = self.args.damage_potential * min(1., row["verified_damage_fraction"])
        reward = local_reward(outcome, prior_potential, potential, PPO_SETTINGS["gamma"])
        row["reward"] += reward
        deaths = self.stage.hits - row["before_lives_lost"]
        if deaths > row["section_deaths"] and row["first_suffix_miss_frame"] is None:
            row["first_suffix_miss_frame"] = self.stage.stage_frame() or self.stage.last_frame
        row["section_deaths"] = deaths
        row["section_bombs_consumed"] = self.stage.bombs_used - row["before_bombs_used"]
        row["stage_frame"] = self.stage.last_frame
        row["last_action"] = action.tolist()
        self.step_seconds += perf_counter() - started
        if terminated or truncated:
            terminal = resources(self.stage)
            row.update(outcome=outcome, terminal_resources=terminal, native_stage=info["native_stage"],
                       native_stage_frame=self.stage.stage_frame(), zero_continue=True,
                       stage2_lives=terminal["lives"] if outcome == "clear" else None,
                       stage2_bombs=terminal["bombs"] if outcome == "clear" else None)
            self.stage.release()
            self.log["episodes"].append(row)
            self.log["current"] = None
        if terminated or truncated or row["section_decisions"] % 128 == 0:
            self.save_report()
        result = {"outcome": outcome, "verified_damage": damage}
        if terminated or truncated:
            result.update(credit_result=row)
        return action_observation(self.student, self.stage, observation), reward, terminated, truncated, result


class SaveUpdates(BaseCallback):
    """Persist PPO and optimizer only after a completed optimization update."""

    def __init__(self, env, log, storage, expected):
        super().__init__()
        self.rollout = env
        self.log = log
        self.storage = storage
        self.expected = expected
        self.saved_steps = log["starting_steps"]

    def save_update(self):
        if self.num_timesteps <= self.saved_steps:
            return
        checkpoints = self.storage.save(self.model.save, dict(
            steps=self.num_timesteps, compatibility=self.expected, warm_start_actor_exact=True,
            resume_checkpoint_sha256=self.log["resume_checkpoint_sha256"],
            prefix_attempt_cursor=self.rollout.prefix_cursor,
        ))
        self.log["checkpoint_storage"].update(checkpoints)
        self.log["checkpoints"]["ppo_sha256"] = checkpoints["current"]["sha256"]
        self.log["checkpoint_steps"] = self.num_timesteps
        update = dict(steps=self.num_timesteps, optimization_epochs=self.model._n_updates,
                      accepted_episodes=len(self.log["episodes"]), prefix_attempts=len(self.log["prefix_attempts"]),
                      stage2_entries=sum(episode["outcome"] == "clear" for episode in self.log["episodes"]))
        update.update({key.removeprefix("train/"): float(value) for key, value in self.logger.name_to_value.items()
                       if key.startswith("train/") and isfinite(value)})
        self.log["updates"].append(update)
        self.saved_steps = self.num_timesteps
        self.rollout.save_report()
        print(json.dumps(dict(status="update", **update)), flush=True)

    def _on_rollout_start(self):
        self.save_update()

    def _on_step(self):
        return True

    def _on_training_end(self):
        self.save_update()


def comparable_observations(student, template):
    return (student.history == template.history
            and student.policy.observation_space == template.policy.observation_space
            and student.policy.action_space == template.policy.action_space
            and all(getattr(student.policy, field) == getattr(template.policy, field)
                    for field in ("motion_spans", "retain_recent_motion", "stage_clock")))


def specification(args, prefix, student, template):
    reward = dict(stage2_entry=1., native_game_over=0., ordinary_transition=0.,
                  timeout_is_truncation=True, damage_potential=args.damage_potential,
                  damage_exclusion="same vulnerable boss and spell phase with HP drop in 1–512",
                  shaping="gamma * successor_potential - previous_potential; no terminal potential")
    return dict(base_sha256=template.provenance["weights_sha256"],
                initialization_sha256=student.provenance["weights_sha256"],
                train_limit=args.section_limit, seed=args.seed, ppo_hyperparameters=PPO_SETTINGS,
                reward=reward, task=dict(title="TH14", version="1.00b", stage=1, difficulty="Normal", shot="Reimu A",
                                         start_frame=args.start_frame, handoff_frame=args.handoff_frame,
                                         allowed_handoff_overshoot=8, stage2_only_success=True,
                                         flags=dict(geometry=False, aim_correction=False, automatic_bomb=False,
                                                    full_native_lives=True, continue_allowed=False,
                                                    native_game_rng_controlled=False)),
                prefix=dict(provenance=prefix.provenance, policy_seed_rotation=args.prefix_seeds))


def train(args, prefix, student, expected, resume, *, rollout_class=BossRollout, stage_session=native_stage,
          extra_sources=()):
    started = perf_counter()
    previous_manifest, previous_hash = resume if resume else ({}, None)
    sources = (evaluate_bomb_crossover, evaluate_full_credit, full_credit_reset, stage_policy, train_full_policy,
               *extra_sources)
    log = dict(status="started", specification=expected, starting_steps=previous_manifest.get("steps", 0),
               requested_additional_steps=args.steps, resume_checkpoint_sha256=previous_hash,
               checkpoints=dict(base_sha256=expected["base_sha256"],
                                initialization_sha256=expected["initialization_sha256"]),
               checkpoint_storage=dict(budget_bytes=args.checkpoint_budget_mib * MIB,
                                       reserve_bytes=args.disk_reserve_mib * MIB),
               prefix_attempt_cursor=previous_manifest.get("prefix_attempt_cursor", 0),
               runtime=dict(runner_sha256=digest(Path(__file__)),
                            source_sha256={module.__name__: digest(Path(module.__file__)) for module in sources},
                            parakit_offsets_sha256=digest(args.parakit / "offsets.py"),
                            torch=str(torch.__version__), stable_baselines3=evaluate_full_credit.evaluate_sft_native.sb3_version),
               prefix_attempts=[], episodes=[], updates=[], current=None)
    publish_json(args.out / "report.json", log)
    try:
        with stage_session(args, {"prefix": prefix, "student": student}) as stage:
            env = rollout_class(stage, prefix, student, args, log, log["prefix_attempt_cursor"])
            storage = CheckpointStore(args.out, args.checkpoint_budget_mib * MIB, args.disk_reserve_mib * MIB)
            model = PPO.load(args.resume, env=env, device="cpu") if resume else warm_start(student.policy, env, args.seed)
            model.gamma = PPO_SETTINGS["gamma"]
            model.rollout_buffer.gamma = model.gamma
            if model.num_timesteps != log["starting_steps"]:
                raise RuntimeError("The native PPO checkpoint step count differs from its validated manifest.")
            logger = configure(str(args.out), ["stdout", "csv"])
            try:
                model.set_logger(logger)
                callback = SaveUpdates(env, log, storage, expected)
                model.learn(total_timesteps=args.steps, callback=callback, reset_num_timesteps=not bool(resume))
                if callback.saved_steps != log["starting_steps"] + args.steps:
                    raise RuntimeError("PPO did not durably save the requested number of suffix decisions.")
            finally:
                logger.close()
        log["status"] = "complete"
    except BaseException as error:
        log.update(status="interrupted", error=dict(type=type(error).__name__, detail=str(error)))
        raise
    finally:
        log["seconds"] = perf_counter() - started
        publish_json(args.out / "report.json", log)
    outcomes = Counter(row["outcome"] for row in log["episodes"])
    print(json.dumps(dict(status="complete", checkpoint_steps=log["checkpoint_steps"],
                          episode_outcomes=dict(outcomes))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo211", type=Path, default=PPO211)
    parser.add_argument("--ppo397", type=Path, default=PPO397)
    parser.add_argument("--ppo-template", type=Path, default=TEMPLATE)
    initial = parser.add_mutually_exclusive_group()
    initial.add_argument("--init-policy", type=Path, help="Single-network state dict with SFT-format sidecar.")
    initial.add_argument("--init-ppo", type=Path, help="Initial PPO archive; copies the actor, reinitializes critic.")
    parser.add_argument("--steps", type=int, default=1024, help="Additional learner-only decisions in whole PPO updates.")
    parser.add_argument("--handoff-frame", type=int, default=7600)
    parser.add_argument("--section-limit", type=int, default=2400,
                        help="Maximum learner decisions per credit, independent of optimizer batch size.")
    parser.add_argument("--start-frame", type=int, default=30)
    parser.add_argument("--prefix-seeds", nargs="+", type=int, default=[1401, 1402, 1501, 1502])
    parser.add_argument("--damage-potential", type=float, default=.15)
    parser.add_argument("--seed", type=int, default=821)
    parser.add_argument("--focus-wait", type=int, default=30)
    parser.add_argument("--checkpoint-budget-mib", type=int, default=64)
    parser.add_argument("--disk-reserve-mib", type=int, default=20480)
    parser.add_argument("--parakit", type=Path, default=Path(__file__).resolve().parents[2] / "parakit")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("The native TH14 1.00b game adapter requires Windows.")
    if (args.steps < 1 or args.steps % PPO_SETTINGS["n_steps"] or args.section_limit < 1
            or not 0 < args.start_frame < args.handoff_frame or args.focus_wait < 0
            or args.checkpoint_budget_mib < 1 or args.disk_reserve_mib < 0):
        parser.error("Use whole PPO updates, valid positive native limits, and a usable checkpoint budget.")
    if not 0 <= args.damage_potential <= .25:
        parser.error("Use a damage potential between zero and 0.25.")
    if (not 0 <= args.seed < 2**32 or len(set(args.prefix_seeds)) != len(args.prefix_seeds)
            or any(not 0 <= seed < 2**63 for seed in args.prefix_seeds)):
        parser.error("Use a valid training seed and distinct prefix policy seeds.")
    torch.set_num_threads(1)
    args.limit = args.handoff_frame + args.section_limit + 8
    try:
        arms = crossover_arms(args)
        prefix = arms[SPLICE]
        template = load_arm(args.ppo_template)
        student = (load_arm(args.init_policy) if args.init_policy else
                   load_ppo_arm(args.init_ppo, template) if args.init_ppo else arms["ppo397"])
        if not comparable_observations(student, template):
            raise ValueError("The initial actor must use the same observation and action transformations as the template.")
        expected = specification(args, prefix, student, template)
        resume = None
        if args.resume:
            manifest = checkpoint_manifest(args.resume)
            if not manifest or "prefix_attempt_cursor" not in manifest:
                raise ValueError("This resume path is not a previous native boss-suffix checkpoint.")
            fingerprint, actor_exact, _ = read_resume(args.resume, expected)
            if not actor_exact:
                raise ValueError("The native suffix lineage lacks an exactly matched warm-start actor.")
            resume = (manifest, fingerprint)
        args.out.mkdir(parents=True, exist_ok=False)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    with output_lock(args.out):
        train(args, prefix, student, expected, resume)


if __name__ == "__main__":
    main()
