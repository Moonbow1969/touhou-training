import argparse
from collections import deque
import json
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import torch

from environment.structured import StateReader, load_parakit_offsets
from scripts.pretrain_policy import (MOTION_INTERVALS, action_space, append_motion_history,
                                    condition_context, network_kwargs, observation_space)


class NormalStageOne(gym.Env):
    def __init__(self, parakit_path, limit=3000, max_misses=1, motion_history=False, history_frames=12,
                 last_stage=1, first_stage=1, prepared=False, stage_practice=False):
        from environment import interface as game

        self.game = game
        self.reader = StateReader(game, load_parakit_offsets(parakit_path), 96)
        self.observation_space = observation_space(motion_history)
        self.motion_history = motion_history
        self.history = deque(maxlen=max(MOTION_INTERVALS + 1, history_frames + 1))
        self.action_space = action_space()
        self.limit = limit
        self.max_misses = max_misses
        self.first_stage = first_stage
        self.last_stage = last_stage
        self.prepared = prepared
        self.stage_practice = stage_practice
        self.held = set()
        self.reset_count = 0
        self.completed = []
        game._get_focus()
        if game.read_game_val("game_state") != 2:
            raise RuntimeError("Start a TH14 Normal game before running this command.")
        self.freeze()
        try:
            self.snapshot = self.reader.snapshot()
            self.require_task(require_start_stage=False)
        except BaseException:
            game.resume_game_process()
            raise

    def freeze(self):
        if not self.game.suspend_game_process():
            raise RuntimeError("Could not lock the TH14 training process.")

    def require_task(self, require_start_stage=True):
        offsets = self.reader.offsets.environment
        character = self.reader.read_uint(offsets.character, relative=True)
        subshot = self.reader.read_uint(offsets.subshot, relative=True)
        if ((self.snapshot.difficulty, character, subshot) != (1, 0, 0)
                or require_start_stage and self.snapshot.stage != self.first_stage):
            raise RuntimeError(f"Expected TH14 Normal Stage {self.first_stage} with Reimu A.")

    def current_stage(self):
        return self.reader.read_uint(self.reader.offsets.environment.stage, relative=True)

    def current_screen(self):
        return self.reader.read_uint(self.reader.offsets.statics.game_screen, relative=True)

    def practice_cleared(self, stage_number, state):
        return (self.stage_practice and state == 1 and stage_number == self.first_stage
                and self.current_screen() == 7 and self.game.read_game_val("lives") >= 0)

    def resolve_transition(self):
        game = self.game
        stage = self.current_stage()
        state = game.read_game_val("game_state")
        if state == 0 and stage <= self.last_stage:
            # The native stage-clear results temporarily set the pause flag.
            self.release()
            for _ in range(24):
                game.require_game_focus()
                game._sleep(20)
                self.freeze()
                stage = self.current_stage()
                state = game.read_game_val("game_state")
                if (stage > self.last_stage or state != 0
                        or self.last_stage == 6 and self.current_screen() == 15):
                    break
        return stage, state

    def stage_frame(self):
        thread = self.reader.offsets.game_thread
        pointer = self.reader.read_uint(thread.game_thread_pointer, relative=True)
        return self.reader.read_uint(pointer + thread.zGameThread_stage_timer) if pointer else 0

    def release(self):
        for key in self.held:
            self.game.keyboard.release(key)
        self.held.clear()

    def observation(self, motion_spans=None, retain_recent_motion=False):
        context = self.current_context()
        if self.motion_history:
            recent = list(self.history)
            if motion_spans is None:
                recent = recent[-(MOTION_INTERVALS + 1):]
            frames, contexts = zip(*recent)
            context = append_motion_history(np.stack(contexts), np.asarray(frames), motion_spans,
                                            retain_recent_motion)[-1]
        return {
            "bullets": self.snapshot.observation["bullets"],
            "context": context,
        }

    def current_context(self):
        return condition_context(self.snapshot.observation["context"], 1, "ReimuA")

    def remember(self):
        if self.motion_history:
            self.history.append((self.last_frame, self.current_context()))

    def retry(self):
        game = self.game
        stage_number, state = self.resolve_transition()
        if state != 2:
            # When the verified stage marker changes first, let the next stage load before Retry.
            for _ in range(24):
                game.require_game_focus()
                game._sleep(20)
                self.freeze()
                state = game.read_game_val("game_state")
                if state != 0:
                    break
        if state != 2:
            raise RuntimeError("The native game is not in an active story stage for Retry.")
        game.resume_game_process()
        if stage_number > self.first_stage or self.current_stage() > self.first_stage:
            # The pause flag returns to active before the new stage accepts its retry shortcut.
            game._sleep(max(0, 90 - self.stage_frame()))
        game.force_reset()
        if self.current_stage() != self.first_stage:
            game.release_all_keys()
            game._sleep(60)
            game.force_reset()
        game.release_all_keys()
        self.freeze()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        game = self.game
        game.require_game_focus()
        self.release()
        if self.reset_count or not self.prepared:
            self.retry()
        self.snapshot = self.reader.snapshot()
        self.require_task()
        self.reset_count += 1
        self.steps = 0
        self.stages_cleared = 0
        self.first_score = self.snapshot.score
        self.last_frame = self.stage_frame()
        self.history.clear()
        self.remember()
        self.hits = 0
        self.first_miss_frame = None
        self.first_miss_stage = None
        self.bombs_used = 0
        self.previous_bombs = game.read_game_val("bombs")
        return self.observation(), {}

    def apply(self, action):
        move, focus, shoot, bomb = map(int, action)
        dy, dx = divmod(move, 3)
        keys = set()
        keys.update(("left",) if dx == 0 else ("right",) if dx == 2 else ())
        keys.update(("up",) if dy == 0 else ("down",) if dy == 2 else ())
        keys.update(key for value, key in ((focus, "shift"), (shoot, "z"), (bomb, "x")) if value)
        if self.game.read_game_val("in_dialog") == -1:
            keys.add("ctrl")
            # Dialogue needs key-down edges even when the shooting policy holds Z or stays idle.
            if "z" in self.held:
                keys.discard("z")
            else:
                keys.add("z")
        for key in self.held - keys:
            self.game.keyboard.release(key)
        for key in keys - self.held:
            self.game.keyboard.press(key)
        self.held = keys

    def step(self, action):
        game = self.game
        game.require_game_focus()
        self.apply(action)
        game._sleep(4)
        self.freeze()
        self.steps += 1
        stage_number, state = self.resolve_transition()
        story_clear = not self.stage_practice and self.last_stage == 6 and self.current_screen() == 15
        practice_clear = self.practice_cleared(stage_number, state)
        cleared = stage_number > self.last_stage or story_clear or practice_clear
        native_end = state != 2 and not cleared
        reward = 0.003
        advanced = (7 if story_clear else stage_number) - self.first_stage + int(practice_clear)
        self.stages_cleared = max(self.stages_cleared, advanced)
        if state == 2:
            past = self.snapshot
            self.snapshot = self.reader.snapshot()
            current = self.snapshot
            if current.stage != past.stage:
                self.history.clear()
                if not cleared:
                    self.last_frame = self.stage_frame()
            elif not cleared:
                self.last_frame = max(self.last_frame, self.stage_frame())
            if cleared:
                self.history.clear()
            self.remember()
            missed = max(0, past.lives - current.lives)
            if missed and self.first_miss_frame is None:
                self.first_miss_frame = self.last_frame
                self.first_miss_stage = current.stage
            self.hits += missed
            bombs = game.read_game_val("bombs")
            used = bombs < self.previous_bombs and not missed
            self.bombs_used += used
            self.previous_bombs = bombs
            reward += np.clip((current.score - past.score) / 30_000, 0, 0.2) - 4 * missed - 0.06 * used
        elif native_end and state == 1 and self.snapshot.lives >= 0 and game.read_game_val("lives") < 0:
            self.hits += 1
            if self.first_miss_frame is None:
                self.first_miss_frame = self.last_frame
                self.first_miss_stage = self.snapshot.stage
        terminated = native_end or cleared or self.hits >= self.max_misses
        truncated = not terminated and self.steps >= self.limit
        reason = ("story_clear" if story_clear else "practice_clear" if practice_clear else "advanced_stage" if cleared
                  else "native_end" if native_end else "miss" if self.hits >= self.max_misses else "limit")
        score = game.read_game_val("score") if cleared else self.snapshot.score
        info = {"stage_frame": self.last_frame, "native_stage": stage_number,
                "native_pause_state": state, "stages_cleared": self.stages_cleared,
                "score": score - self.first_score,
                "misses": int(self.hits), "bombs_used": int(self.bombs_used),
                "steps": self.steps, "first_miss_frame": self.first_miss_frame,
                "first_miss_stage": self.first_miss_stage,
                "outcome": reason if terminated or truncated else None}
        if terminated or truncated:
            self.completed.append(info.copy())
        return self.observation(), float(reward), terminated, truncated, info

    def close(self):
        self.game.resume_game_process()
        self.release()


def run_episode(env, policy):
    observation, _ = env.reset()
    reward_sum = 0.0
    while True:
        observation, reward, terminated, truncated, info = env.step(policy(observation))
        reward_sum += reward
        if terminated or truncated:
            return dict(info, reward=reward_sum)


class StageProgress(BaseCallback):
    def __init__(self, env, out, first_episode, every):
        super().__init__()
        self.stage_env = env
        self.out = out
        self.first_episode = first_episode
        self.every = every
        self.last_saved = 0
        self.started = perf_counter()

    def _on_step(self):
        return True

    def _on_training_start(self):
        self.last_saved = self.num_timesteps

    def _on_rollout_start(self):
        if self.num_timesteps - self.last_saved >= self.every:
            self.save_progress()

    def _on_training_end(self):
        self.save_progress()

    def save_progress(self):
        self.last_saved = self.num_timesteps
        episodes = self.stage_env.completed[self.first_episode:]
        info = {"steps": self.num_timesteps, "updates": self.model._n_updates,
                "seconds": perf_counter() - self.started, "completed_episodes": len(episodes),
                "stage_clears": sum(row["outcome"] == "advanced_stage" for row in episodes),
                "best_stage_frame": max((row["stage_frame"] for row in episodes), default=0)}
        checkpoint = self.out / "training-checkpoint.tmp.zip"
        self.model.save(checkpoint)
        checkpoint.replace(self.out / "training-checkpoint.zip")
        temporary = self.out / "training-progress.tmp"
        temporary.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.out / "training-progress.json")
        print("progress", json.dumps(info), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Stage-one gameplay from a cloned policy, then PPO.")
    initial = parser.add_mutually_exclusive_group(required=True)
    initial.add_argument("--bc", type=Path)
    initial.add_argument("--resume-ppo", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parakit", type=Path, default=Path(__file__).resolve().parents[2] / "parakit")
    parser.add_argument("--rl-steps", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--motion-history", action="store_true")
    parser.add_argument("--max-misses", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=63)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.attempts, args.limit, args.checkpoint_every) < 1 or args.rl_steps < 0 or not 1 <= args.max_misses <= 2:
        parser.error("Use positive attempts and limits, nonnegative RL steps, and one or two misses.")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    env = NormalStageOne(args.parakit, args.limit, args.max_misses, args.motion_history)
    report = {}
    try:
        if args.resume_ppo:
            model = PPO.load(args.resume_ppo, env=env, device=args.device)
        else:
            model = PPO("MultiInputPolicy", env, n_steps=256, batch_size=64, n_epochs=4,
                        learning_rate=8e-5, gamma=0.997, clip_range=0.12, ent_coef=0.003,
                        policy_kwargs=network_kwargs(args.hidden), device=args.device, seed=args.seed)
            model.policy.load_state_dict(torch.load(args.bc, map_location=model.device, weights_only=True))
        strategies = {"bc": lambda obs: model.predict(obs, deterministic=True)[0],
                      "stationary": lambda _: np.array([4, 0, 1, 0])}
        for key, strategy in (() if args.skip_baselines else strategies.items()):
            report[key] = []
            for _ in range(args.attempts):
                result = run_episode(env, strategy)
                report[key].append(result)
                print(key, json.dumps(result), flush=True)
                (args.out / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if args.rl_steps:
            start = perf_counter()
            first = len(env.completed)
            progress = StageProgress(env, args.out, first, args.checkpoint_every)
            model.learn(total_timesteps=args.rl_steps, callback=progress, reset_num_timesteps=not args.resume_ppo)
            model.save(args.out / "ppo")
            report["rl_training"] = {"seconds": perf_counter() - start, "steps": model.num_timesteps,
                                     "updates": model._n_updates, "episodes": env.completed[first:]}
            report["bc_ppo"] = []
            (args.out / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            for _ in range(args.attempts):
                result = run_episode(env, strategies["bc"])
                report["bc_ppo"].append(result)
                print("bc_ppo", json.dumps(result), flush=True)
                (args.out / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    finally:
        env.close()


if __name__ == "__main__":
    main()
