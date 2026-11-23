import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from torch import nn

from environment.structured import StructuredTouhou14Env
from scripts.train_structured import evaluate


def projectile_features(bullets):
    position = bullets[:, :, :2] * bullets.new_tensor((384, 448))
    velocity = bullets[:, :, 2:4] * 16
    closest_time = (-(position * velocity).sum(dim=-1, keepdim=True)
                    / velocity.square().sum(dim=-1, keepdim=True).clamp(min=0.1)).clamp(0, 24)
    radius = bullets[:, :, 4:5] * 32
    nearest = position + velocity * closest_time
    clearance = torch.linalg.vector_norm(nearest, dim=-1, keepdim=True) - radius
    distance = torch.linalg.vector_norm(position, dim=-1, keepdim=True) - radius
    future = (position + 16 * velocity) / 192
    return torch.cat((
        bullets[:, :, :6], closest_time / 24,
        (distance / 128).clamp(-1, 1), (clearance / 128).clamp(-1, 1),
        future.clamp(-2, 2),
    ), dim=-1)


class ThreatEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=192 + observation_space["context"].shape[0])
        self.encode = nn.Sequential(nn.Linear(11, 96), nn.ReLU(), nn.Linear(96, 96), nn.ReLU())

    def forward(self, observations):
        bullets = observations["bullets"]
        mask = bullets[:, :, 6:]
        embeddings = self.encode(projectile_features(bullets)) * mask
        mean = embeddings.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return torch.cat((embeddings.amax(dim=1), mean, observations["context"]), dim=1)


class AttentiveThreatEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        context_size = observation_space["context"].shape[0]
        super().__init__(observation_space, features_dim=192 + context_size)
        self.entities = nn.Sequential(nn.Linear(11, 96), nn.SiLU(), nn.Linear(96, 64))
        self.queries = nn.Linear(context_size, 4 * 32)
        self.keys = nn.Linear(64, 32)
        self.values = nn.Linear(64, 32)
        self.register_buffer("attention_scale", torch.tensor(32 ** -.5), persistent=False)

    def forward(self, observations):
        bullets, context = observations["bullets"], observations["context"]
        mask = (bullets[..., 6] > 0) & (bullets[..., 5] > 0)
        entities = self.entities(projectile_features(bullets))
        query = self.queries(context).reshape(-1, 4, 32)
        logits = torch.bmm(query, self.keys(entities).transpose(1, 2)) * self.attention_scale
        weights = logits.masked_fill(~mask.unsqueeze(1), -1e4).softmax(dim=-1) * mask.unsqueeze(1)
        pooled = torch.bmm(weights, self.values(entities)).flatten(1)
        summary = (entities * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return torch.cat((pooled, summary, context), dim=1)


class TimedQRDQN(QRDQN):
    training_seconds = 0.0

    def train(self, gradient_steps, batch_size=100):
        start = perf_counter()
        super().train(gradient_steps, batch_size)
        self.training_seconds += perf_counter() - start


class SpellReplayBuffer(DictReplayBuffer):
    def __init__(self, *args, success_fraction=0.35, **kwargs):
        super().__init__(*args, **kwargs)
        if self.n_envs != 1 or not 0 < success_fraction < 1:
            raise ValueError("Capture replay requires one environment and a fraction in (0, 1).")
        self.success_fraction = success_fraction
        self.success = np.zeros(self.buffer_size, dtype=bool)
        self.episode_indices = []

    def add(self, obs, next_obs, action, reward, done, infos):
        index = self.pos
        super().add(obs, next_obs, action, reward, done, infos)
        self.success[index] = False
        self.episode_indices.append(index)
        if done[0]:
            if infos[0].get("captured"):
                self.success[self.episode_indices] = True
            self.episode_indices.clear()

    def sample(self, batch_size, env=None):
        successful = np.flatnonzero(self.success)
        if not len(successful):
            return super().sample(batch_size, env=env)
        count = round(batch_size * self.success_fraction)
        limit = self.buffer_size if self.full else self.pos
        indices = np.concatenate((
            np.random.choice(successful, count),
            np.random.randint(limit, size=batch_size - count),
        ))
        return self._get_samples(indices, env=env)


class Progress(BaseCallback):
    def __init__(self, env, out, start, initial_steps):
        super().__init__()
        self.game = env
        self.out = out
        self.start = start
        self.initial_steps = initial_steps

    def summary(self):
        return {
            "steps": self.model.num_timesteps - self.initial_steps,
            "total_steps": self.model.num_timesteps,
            "total_updates": self.model._n_updates,
            "elapsed_seconds": perf_counter() - self.start,
            "training_seconds": self.model.training_seconds,
            "game_step_seconds": self.game.play_seconds,
            "decode_seconds": self.game.decode_seconds,
            "reset_seconds": self.game.reset_seconds,
            "game_frames": self.game.actual_frames,
            "episodes": len(self.game.completed),
            "outcomes": dict(Counter(item["outcome"] for item in self.game.completed)),
            "recent_episodes": self.game.completed[-10:],
        }

    def _on_step(self):
        if self.n_calls % 256 == 0:
            summary = self.summary()
            with (self.out / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(summary) + "\n")
            print(f"steps={summary['steps']} episodes={summary['outcomes']} "
                  f"seconds={summary['elapsed_seconds']:.1f}", flush=True)
        return True


def evaluate_policies(env, model, episodes, seed, out):
    rng = np.random.default_rng(seed)
    policies = {
        "qrdqn": lambda obs: int(model.predict(obs, deterministic=True)[0]),
        "stationary": lambda _: 0,
        "random": lambda _: int(rng.integers(env.action_space.n)),
    }
    results = {name: [] for name in policies}
    # Alternate policies so game-clock drift affects each policy throughout the run.
    env.reset()
    for _ in range(episodes):
        for name, policy in policies.items():
            result = evaluate(env, policy, 1)[0]
            results[name].append(result)
            (out / "evaluation.json").write_text(
                json.dumps(results, indent=2) + "\n", encoding="utf-8"
            )
            evidence = out / "capture.png"
            if name == "qrdqn" and result["captured"] and not evidence.exists():
                env.game.resume_game_process()
                env.game.release_all_keys()
                env.game._sleep(18)
                env.freeze()
                window = env.game._game_window
                env.game.pyscreeze.screenshot(
                    region=(window.left, window.top, window.width, window.height)
                ).save(evidence)
            print(f"{name} attempt {len(results[name])}: {result['outcome']}, "
                  f"damage={result['damage']}, frames={result['episode_frames']}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="QR-DQN for a TH14 spell selected in thprac.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parakit", type=Path, default=Path(__file__).resolve().parents[2] / "parakit")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--difficulty", type=int, choices=range(4), default=0)
    parser.add_argument("--spell", type=int, default=3)
    parser.add_argument("--episode-limit", type=int, default=750)
    parser.add_argument("--success-fraction", type=float, default=0.0)
    args = parser.parse_args()
    if args.steps < 0 or args.eval_episodes < 0 or (not args.steps and not args.checkpoint):
        parser.error("Provide positive training steps or a checkpoint for evaluation.")
    if not 0 <= args.success_fraction < 1:
        parser.error("The capture replay fraction must be in [0, 1).")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    env = StructuredTouhou14Env(
        args.parakit, prepared=True, episode_limit=args.episode_limit,
        expect_difficulty=args.difficulty, expect_spell=args.spell,
    )
    try:
        replay = {}
        if args.success_fraction:
            replay = {"replay_buffer_class": SpellReplayBuffer,
                      "replay_buffer_kwargs": {"success_fraction": args.success_fraction}}
        if args.checkpoint:
            model = TimedQRDQN.load(args.checkpoint, env=env, device=args.device,
                                   seed=args.seed, **replay)
        else:
            model = TimedQRDQN(
                "MultiInputPolicy", env, learning_rate=1e-4, buffer_size=20_000,
                learning_starts=256, train_freq=4, batch_size=64, gamma=0.995,
                target_update_interval=600, exploration_fraction=0.45,
                exploration_final_eps=0.08, seed=args.seed, device=args.device,
                policy_kwargs={"features_extractor_class": ThreatEncoder,
                               "net_arch": [128, 128], "n_quantiles": 32},
                **replay,
            )
        model.training_seconds = 0.0
        if args.steps:
            if model.num_timesteps:
                model.learning_starts = model.num_timesteps + 256
            progress = Progress(env, args.out, perf_counter(), model.num_timesteps)
            model.learn(
                total_timesteps=args.steps, reset_num_timesteps=not args.checkpoint,
                callback=[progress, CheckpointCallback(save_freq=1024, save_path=args.out)],
            )
            model.save(args.out / "model")
            summary = progress.summary()
            summary["training_episodes"] = list(env.completed)
            summary["max_bullet_count"] = env.max_bullets
            summary["device"] = str(model.device)
            summary["success_fraction"] = args.success_fraction
            (args.out / "training.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            print("training summary", json.dumps({key: value for key, value in summary.items()
                                                  if not isinstance(value, list)}), flush=True)
        if args.eval_episodes:
            evaluate_policies(env, model, args.eval_episodes, args.seed, args.out)
    finally:
        env.close()


if __name__ == "__main__":
    main()
