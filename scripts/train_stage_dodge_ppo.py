import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.policies import ActorCriticPolicy
import torch

from scripts.evaluate_stage_policies import read_policy
from scripts.probe_stage_risk import player_radius
from scripts.stage_policy import NormalStageOne, StageProgress, run_episode
from scripts.stage_risk import action_clearances, dodge_regret
from scripts.train_stage_dodge import DodgeHead, candidate_features


class DodgeEnv(gym.Wrapper):
    def __init__(self, env, base, risk_weight):
        super().__init__(env)
        self.base = base
        self.risk_weight = risk_weight
        self.observation_space = gym.spaces.Box(-2, 2, (18, 5), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(18)

    def decision_features(self, observation):
        self.base_action = self.base.predict(observation, deterministic=True)[0]
        self.clearances = action_clearances(observation, self.radius)
        return candidate_features(self.base_action, self.clearances)

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self.radius = player_radius(self.env)
        return self.decision_features(observation), info

    def step(self, action):
        buttons = self.base_action.copy()
        focus, direction = divmod(int(action), 9)
        buttons[:2] = (direction, focus)
        risk = dodge_regret(buttons, self.clearances)
        observation, reward, terminated, truncated, info = self.env.step(buttons)
        features = self.decision_features(observation)
        return features, reward - self.risk_weight * risk, terminated, truncated, info


class FlatDodgeHead(DodgeHead):
    def forward(self, features):
        return super().forward(features.reshape(-1, 18, 5))


class DodgeActorCritic(ActorCriticPolicy):
    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.action_net = FlatDodgeHead()
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )


def main():
    parser = argparse.ArgumentParser(description='Fine-tune the imitation dodge head with native rollouts.')
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--head', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--steps', type=int, default=1024)
    parser.add_argument('--limit', type=int, default=1500)
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--risk-weight', type=float, default=.025)
    parser.add_argument('--seed', type=int, default=77)
    args = parser.parse_args()
    if min(args.steps, args.limit, args.attempts) < 1 or args.risk_weight < 0:
        parser.error('Use positive steps, limit and attempts, and a nonnegative risk weight.')
    torch.set_num_threads(1)
    base = read_policy(args.base, torch.device('cuda'), motion_history=True)
    initial = torch.load(args.head, map_location='cpu', weights_only=True)
    args.out.mkdir(parents=True, exist_ok=False)
    stage = NormalStageOne(Path(__file__).resolve().parents[2] / 'parakit', args.limit,
                           max_misses=1, motion_history=True)
    try:
        env = DodgeEnv(stage, base, args.risk_weight)
        model = PPO(
            DodgeActorCritic, env, device='cpu', seed=args.seed, n_steps=256, batch_size=64,
            n_epochs=4, learning_rate=8e-5, gamma=.997, clip_range=.15, ent_coef=.004,
            policy_kwargs={'net_arch': {'pi': [], 'vf': [64, 64]}},
        )
        model.policy.action_net.load_state_dict(initial)
        model.set_logger(configure(str(args.out), ['stdout', 'csv']))
        progress = StageProgress(stage, args.out, len(stage.completed), every=256)
        started = perf_counter()
        model.learn(args.steps, callback=progress)
        model.save(args.out / 'ppo')
        fixed = DodgeHead().eval()
        fixed.load_state_dict(initial)

        def imitation(observation):
            with torch.inference_mode():
                return int(fixed(torch.from_numpy(observation)).argmax())

        def fine_tuned(observation):
            return int(model.predict(observation, deterministic=True)[0])

        results = {'imitation': [], 'ppo': []}
        report = {
            'base': str(args.base), 'head': str(args.head), 'limit': args.limit, 'seed': args.seed,
            'risk_weight': args.risk_weight, 'rollout_steps': model.num_timesteps,
            'optimizer_epochs': model._n_updates, 'training_seconds': perf_counter() - started,
            'training_episodes': list(stage.completed), 'results': results,
        }
        for block in range(args.attempts):
            pair = [('imitation', imitation), ('ppo', fine_tuned)]
            for name, strategy in pair if block % 2 == 0 else reversed(pair):
                result = dict(run_episode(env, strategy), block=block + 1)
                results[name].append(result)
                report['summary'] = {
                    label: {'n': len(rows), 'median_furthest_frame': median(r['stage_frame'] for r in rows),
                            'first_miss_frames': [r['first_miss_frame'] for r in rows],
                            'stage_clears': sum(r['outcome'] == 'advanced_stage' for r in rows),
                            'no_miss_at_cutoff': sum(r['outcome'] == 'limit' for r in rows)}
                    for label, rows in results.items() if rows
                }
                (args.out / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
                print(name, result, flush=True)
    finally:
        stage.close()


if __name__ == '__main__':
    main()
