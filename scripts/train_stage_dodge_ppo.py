import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.policies import ActorCriticPolicy
import torch

from scripts.evaluate_boss_aim import aim_features
from scripts.evaluate_stage_policies import read_policy, stage_action
from scripts.pretrain_policy import LEVELS
from scripts.probe_stage_risk import player_radius
from scripts.stage_policy import NormalStageOne, StageProgress, run_episode
from scripts.record_run import write_json
from scripts.stage_risk import action_clearances, body_clearances, dodge_regret, laser_clearances, suggest_dodge
from scripts.train_stage_dodge import DodgeHead, candidate_features


class DodgeEnv(gym.Wrapper):
    def __init__(self, env, base, risk_weight, body_risk=False, laser_risk=False):
        super().__init__(env)
        self.base = base
        self.risk_weight = risk_weight
        self.body_risk = body_risk
        self.laser_risk = laser_risk
        self.aim = False
        self.observation_space = gym.spaces.Box(-2, 2, (18, 5), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(18)

    def decision_features(self, observation):
        self.current_observation = observation
        self.base_action = stage_action(self.base, self.env, observation)
        self.clearances = action_clearances(observation, self.radius)
        reader = self.env.reader
        if self.body_risk:
            hitboxes = reader.read_enemy_hitboxes()
            self.clearances = np.minimum(self.clearances, body_clearances(observation, self.radius, hitboxes))
        if self.laser_risk:
            segments = reader.read_straight_lasers()
            self.clearances = np.minimum(self.clearances, laser_clearances(observation, self.radius, segments))
        return candidate_features(self.base_action, self.clearances)

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self.radius = player_radius(self.env)
        return self.decision_features(observation), info

    def step(self, action):
        buttons = self.base_action.copy()
        focus, direction = divmod(int(action), 9)
        buttons[:2] = (direction, focus)
        if self.aim:
            _, direction = aim_features(self.current_observation, buttons, self.clearances, self.env.snapshot.boss)
            buttons[0] = direction
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
    parser.add_argument('--difficulty', choices=LEVELS[:4], default='Normal')
    parser.add_argument('--steps', type=int, default=1024)
    parser.add_argument('--limit', type=int, default=1500)
    parser.add_argument('--evaluation-limit', type=int, help='Use a separate decision bound during native evaluation.')
    parser.add_argument('--evaluation-max-misses', type=int, choices=(1, 2), default=1)
    parser.add_argument('--evaluation-aim', action='store_true', help='Use the same geometric boss aim in every evaluation arm.')
    parser.add_argument('--evaluation-reference', type=Path, help='Include a frozen PPO checkpoint in the evaluation.')
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--risk-weight', type=float, default=.025)
    parser.add_argument('--seed', type=int, default=77)
    parser.add_argument('--initial-temperature', type=float, default=1.,
                        help='Rescale the cloned actor logits for rollout exploration at initialization.')
    parser.add_argument('--body-risk', action='store_true')
    parser.add_argument('--laser-risk', action='store_true')
    parser.add_argument('--analytic-control', action='store_true', help='Evaluate the same geometry with rule-based selection.')
    parser.add_argument('--focus-wait', type=int, default=0)
    parser.add_argument('--resume-ppo', type=Path, help='Continue from a saved PPO checkpoint.')
    args = parser.parse_args()
    if (min(args.steps, args.limit, args.attempts) < 1 or args.risk_weight < 0 or args.focus_wait < 0
            or args.evaluation_limit is not None and args.evaluation_limit < 1):
        parser.error('Use positive steps, limit and attempts, and a nonnegative risk weight.')
    if not 0 < args.initial_temperature < float('inf') or args.resume_ppo and args.initial_temperature != 1:
        parser.error('Use a finite positive temperature. Resumed checkpoints keep their actor logits.')
    torch.set_num_threads(1)
    base = read_policy(args.base, torch.device('cuda'), motion_history=True)
    initial = torch.load(args.head, map_location='cpu', weights_only=True)
    args.out.mkdir(parents=True, exist_ok=False)
    stage = NormalStageOne(Path(__file__).resolve().parents[2] / 'parakit', args.limit,
                           max_misses=1, motion_history=True, focus_wait=args.focus_wait,
                           history_frames=max(base.motion_spans or (12,)), difficulty=LEVELS.index(args.difficulty))
    try:
        env = DodgeEnv(stage, base, args.risk_weight, args.body_risk, args.laser_risk)
        if args.resume_ppo:
            model = PPO.load(args.resume_ppo, env=env, device='cpu')
        else:
            model = PPO(
                DodgeActorCritic, env, device='cpu', seed=args.seed, n_steps=256, batch_size=64,
                n_epochs=4, learning_rate=8e-5, gamma=.997, clip_range=.15, ent_coef=.004,
                policy_kwargs={'net_arch': {'pi': [], 'vf': [64, 64]}},
            )
            model.policy.action_net.load_state_dict(initial)
            with torch.no_grad():
                final = model.policy.action_net.score[-1]
                final.weight.div_(args.initial_temperature)
                final.bias.div_(args.initial_temperature)
        model.set_logger(configure(str(args.out), ['stdout', 'csv']))
        progress = StageProgress(stage, args.out, len(stage.completed), every=256)
        started = perf_counter()
        model.learn(args.steps, callback=progress, reset_num_timesteps=not args.resume_ppo)
        model.save(args.out / 'ppo')
        fixed = DodgeHead().eval()
        fixed.load_state_dict(initial)

        def imitation(observation):
            with torch.inference_mode():
                return int(fixed(torch.from_numpy(observation)).argmax())

        def fine_tuned(observation):
            return int(model.predict(observation, deterministic=True)[0])

        def analytic(_):
            action, _, _ = suggest_dodge(env.base_action, env.clearances)
            direction, focus = map(int, action[:2])
            return focus * 9 + direction

        arms = [('imitation', imitation), ('ppo', fine_tuned)]
        if args.analytic_control:
            arms.append(('analytic', analytic))
        if args.evaluation_reference:
            reference = PPO.load(args.evaluation_reference, device='cpu')
            arms.append(('reference_ppo', lambda observation: int(reference.predict(observation, deterministic=True)[0])))
        results = {label: [] for label, _ in arms}
        report = {
            'base': str(args.base), 'head': str(args.head), 'limit': args.limit, 'seed': args.seed,
            'difficulty': args.difficulty,
            'evaluation_limit': args.evaluation_limit or args.limit, 'body_risk': args.body_risk,
            'evaluation_max_misses': args.evaluation_max_misses, 'evaluation_aim': args.evaluation_aim,
            'evaluation_reference': str(args.evaluation_reference) if args.evaluation_reference else None,
            'laser_risk': args.laser_risk, 'resume_ppo': str(args.resume_ppo) if args.resume_ppo else None,
            'initial_temperature': args.initial_temperature,
            'risk_weight': args.risk_weight, 'rollout_steps': model.num_timesteps,
            'optimizer_epochs': model._n_updates, 'training_seconds': perf_counter() - started,
            'training_episodes': list(stage.completed), 'results': results,
        }
        stage.limit = report['evaluation_limit']
        stage.max_misses = args.evaluation_max_misses
        env.aim = args.evaluation_aim
        write_json(args.out / 'evaluation.json', report)
        for block in range(args.attempts):
            rotation = block % len(arms)
            for name, strategy in arms[rotation:] + arms[:rotation]:
                result = dict(run_episode(env, strategy), block=block + 1)
                results[name].append(result)
                report['summary'] = {
                    label: {'n': len(rows), 'median_furthest_frame': median(r['stage_frame'] for r in rows),
                            'median_score': median(r['score'] for r in rows),
                            'median_misses': median(r['misses'] for r in rows),
                            'first_miss_frames': [r['first_miss_frame'] for r in rows],
                            'stage_clears': sum(r['outcome'] == 'advanced_stage' for r in rows),
                            'no_miss_at_cutoff': sum(r['outcome'] == 'limit' for r in rows)}
                    for label, rows in results.items() if rows
                }
                write_json(args.out / 'evaluation.json', report)
                print(name, result, flush=True)
    finally:
        stage.close()


if __name__ == '__main__':
    main()
