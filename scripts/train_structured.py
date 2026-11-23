import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
from torch import nn

from environment.structured import StructuredTouhou14Env


class BulletSetEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=138)
        self.encode = nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())

    def forward(self, observations):
        bullets = observations["bullets"]
        mask = bullets[:, :, 6:]
        features = self.encode(bullets[:, :, :6]) * mask
        mean = features.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return torch.cat((features.amax(dim=1), mean, observations["context"]), dim=1)


def evaluate(env, choose_action, episodes):
    outcomes = []
    for _ in range(episodes):
        observation, _ = env.reset()
        total_reward = 0.0
        while True:
            observation, reward, terminated, truncated, info = env.step(choose_action(observation))
            total_reward += reward
            if terminated or truncated:
                outcomes.append(dict(info, reward=total_reward))
                break
    return outcomes


def main():
    parser = argparse.ArgumentParser(description="Train a screenshot-free TH14 practice agent.")
    parser.add_argument("--parakit", type=Path, default=Path(__file__).resolve().parents[2] / "parakit")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=768)
    parser.add_argument("--eval-episodes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--action-frames", type=int, default=4)
    parser.add_argument("--episode-limit", type=int, default=1000)
    parser.add_argument("--prepared", action="store_true", help="Use the practice already running in TH14.")
    args = parser.parse_args()
    if args.steps <= 96 or args.eval_episodes < 0:
        parser.error("Use at least 97 steps and a non-negative evaluation episode count.")
    args.out.mkdir(parents=True, exist_ok=False)
    env = StructuredTouhou14Env(
        args.parakit, prepared=args.prepared, action_frames=args.action_frames,
        episode_limit=args.episode_limit,
    )
    try:
        start = perf_counter()
        model = DQN(
            "MultiInputPolicy", env, learning_rate=1e-4, buffer_size=8192,
            learning_starts=96, train_freq=4, batch_size=64,
            target_update_interval=500, exploration_fraction=0.65, exploration_final_eps=0.05,
            seed=args.seed, device=args.device, verbose=1,
            policy_kwargs={"features_extractor_class": BulletSetEncoder, "net_arch": [128, 128]},
        )
        model.learn(total_timesteps=args.steps, log_interval=1)
        elapsed = perf_counter() - start
        model.save(args.out / "model")
        summary = {
            "steps": model.num_timesteps,
            "updates": model._n_updates,
            "device": str(model.device),
            "action_frames": args.action_frames,
            "episode_limit": args.episode_limit,
            "parameters": sum(parameter.numel() for parameter in model.policy.q_net.parameters()),
            "elapsed_seconds": elapsed,
            "decisions_per_second": model.num_timesteps / elapsed,
            "max_bullet_count": env.max_bullets,
            "resets": env.reset_count,
            "training_episodes": list(env.completed),
        }
        reloaded = DQN.load(args.out / "model.zip", device=model.device)
        action, _ = reloaded.predict(env.state.observation, deterministic=True)
        assert env.action_space.contains(int(action))
        summary["reloaded_action"] = int(action)
        if args.eval_episodes:
            rng = np.random.default_rng(args.seed)
            summary["random_eval"] = evaluate(
                env, lambda _: int(rng.integers(env.action_space.n)), args.eval_episodes
            )
            summary["dqn_eval"] = evaluate(
                env, lambda obs: int(reloaded.predict(obs, deterministic=True)[0]), args.eval_episodes
            )
        (args.out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
