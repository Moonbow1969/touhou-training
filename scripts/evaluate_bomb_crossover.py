"""Rotate the frozen 16k PPO controls and their Bomb-head crossover on full stock credits."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.distributions import MultiCategoricalDistribution

from scripts import evaluate_full_credit as credit
from scripts.checkpoint_storage import digest
from scripts.evaluate_sft_native import Arm, load_arms, output_lock
from scripts.pretrain_policy import ACTION_WIDTHS, predict_heads, read_shard


REPO = Path(__file__).resolve().parents[1]
DATA = REPO.parents[1] / "outputs/touhou-sft-world"
PPO211 = DATA / "sft-native-data-261123/legacy-sft-to-ppo-long-16384-seed211-fresh/policy.zip"
PPO397 = DATA / "th14-stock-warm-seed397-16384/policy.zip"
TEMPLATE = DATA / "overnight-title-equal-61/sft/policy.pt"
DEV = DATA / "sft-native-data-261123/frozen-training-snapshot/th14_ud06d6-stage1.npz"
HASHES = {"ppo211":"d387a429bca1934b14a313e46101c91d4437556ea1445e46e567fa80c4717d19",
          "ppo397":"4025619296fe7b377ddbc3633ea59dcf413d5b7eb7e5b5f8f2aac20fe9054d36"}
SPLICE = "ppo397_with_ppo211_bomb"


class BombCrossover:
    def __init__(self, actor, bomb):
        if actor.history != bomb.history or actor.policy.observation_space != bomb.policy.observation_space:
            raise ValueError("The two sources must receive identical transformed observations.")
        if actor.policy.action_space != bomb.policy.action_space:
            raise ValueError("The two learned action spaces differ.")
        self.actor, self.bomb = actor.policy, bomb.policy
        for attr in ("motion_spans", "retain_recent_motion", "stage_clock"):
            value = getattr(self.actor, attr)
            if value != getattr(self.bomb, attr):
                raise ValueError(f"Observation transform mismatch in {attr}.")
            setattr(self, attr, value)

    @torch.inference_mode()
    def distribution(self, tensors):
        bullets, context = tensors["bullets"], tensors["context"]
        movement, focus, shoot, _ = predict_heads(self.actor, bullets, context)
        bomb = predict_heads(self.bomb, bullets, context)[3]
        logits = torch.cat((movement, focus, shoot, bomb), dim=-1)
        return MultiCategoricalDistribution(list(ACTION_WIDTHS)).proba_distribution(logits)

    @torch.inference_mode()
    def predict(self, observation, deterministic=False):
        tensors, batched = self.actor.obs_to_tensor(observation)
        actions = self.distribution(tensors).get_actions(deterministic=deterministic).cpu().numpy()
        return (actions if batched else actions[0]), None


def crossover_arms(args):
    arms = load_arms([], [f"ppo397={args.ppo397}", f"ppo211={args.ppo211}"], args.ppo_template)
    if any(arm.provenance["weights_sha256"] != HASHES[name] for name, arm in arms.items()):
        raise ValueError("Checkpoints must match the published 16,384-decision source fingerprints.")
    actor, bomb = arms["ppo397"], arms["ppo211"]
    origin = {"kind":"frozen_logit_crossover", "movement_focus_shoot":"ppo397", "bomb":"ppo211",
              "source_sha256":HASHES, "stage_clock":actor.policy.stage_clock,
              "native_training_decisions_after_composition":0}
    arms[SPLICE] = Arm(BombCrossover(actor, bomb), actor.history, origin)
    return arms


@torch.inference_mode()
def verify(arms):
    meta = json.loads(DEV.with_suffix(".json").read_text(encoding="utf-8"))
    if meta["split"] != "validation" or meta["replay"] != "th14_ud06d6.rpy":
        raise ValueError("Expected the previously used TH14 development replay.")
    bullets, context, _ = read_shard((DEV, dict(meta, samples=meta["pairs"])), motion_history=True, stage_clock=True)
    sources = [arms["ppo397"].policy, arms["ppo211"].policy]
    self_compositions = [BombCrossover(arms[n], arms[n]) for n in ("ppo397", "ppo211")]
    candidate = arms[SPLICE].policy
    largest_difference = 0.
    bomb_differences = []
    for start in range(0, len(bullets), 128):
        obs = {"context":context[start:start+128], "bullets":bullets[start:start+128]}
        tensors, _ = sources[0].obs_to_tensor(obs)
        original = [[h.probs for h in source.get_distribution(tensors).distribution] for source in sources]
        expected = [original[0], original[1], original[0][:3] + original[1][3:]]
        for policy, reference in zip([*self_compositions, candidate], expected):
            for new, target in zip(policy.distribution(tensors).distribution, reference):
                difference = float((new.probs - target).abs().max())
                largest_difference = max(largest_difference, difference)
                if not torch.equal(new.probs, target):
                    raise ValueError(f"Source head probabilities differ (max abs {difference}).")
        bomb_differences.append(original[1][3][:, 1] - original[0][3][:, 1])

    # Match the native single-observation, four categorical draws, including RNG consumption.
    rng_matches, changed_bombs = 0, 0
    indices = np.linspace(0, len(bullets)-1, 128, dtype=int)
    for seed in (1401, 1402, 1403):
        action_streams = []
        for policy in [*sources, *self_compositions, candidate]:
            torch.manual_seed(seed)
            actions = [policy.predict({"bullets":bullets[i], "context":context[i]}, deterministic=False)[0] for i in indices]
            action_streams.append((np.stack(actions), torch.random.get_rng_state()))
        a, b, aa, bb, ab = action_streams
        if not (np.array_equal(a[0], aa[0]) and np.array_equal(b[0], bb[0])
                and np.array_equal(a[0][:, :3], ab[0][:, :3]) and np.array_equal(b[0][:, 3], ab[0][:, 3])):
            raise ValueError("Four-head sampled streams failed the fixed-seed validation.")
        if not all(torch.equal(a[1], state) for _, state in action_streams):
            raise ValueError("Sampling consumed a different amount of randomness.")
        rng_matches += len(indices)
        changed_bombs += int((a[0][:, 3] != ab[0][:, 3]).sum())
    delta = torch.cat(bomb_differences)
    return {"replay":meta["replay"], "split":"development", "npz_sha256":digest(DEV), "observations":len(context),
            "source_sha256":HASHES, "self_and_cross_probabilities_exact":True,
            "max_absolute_probability_difference_vs_intended_heads":largest_difference,
            "single_observation_sampling_seeds":[1401,1402,1403], "single_observation_draws_per_policy":rng_matches,
            "source_and_self_sampled_actions_exact":True, "sampled_crossover_matches_intended_sources":True,
            "rng_states_equal":True, "changed_sampled_bombs_vs_ppo397":changed_bombs,
            "mean_ppo211_minus_ppo397_bomb_probability":float(delta.mean()), "runner_sha256":digest(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppo211", type=Path, default=PPO211)
    parser.add_argument("--ppo397", type=Path, default=PPO397)
    parser.add_argument("--ppo-template", type=Path, default=TEMPLATE)
    parser.add_argument("--verify-only", action="store_true", help="Check archived observations without opening the game.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1401, 1407)))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--parakit", type=Path, default=REPO.parent / "parakit")
    parser.add_argument("--limit", type=int, default=3300)
    parser.add_argument("--start-frame", type=int, default=30)
    parser.add_argument("--focus-wait", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit < 1 or args.start_frame < 1 or args.focus_wait < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use valid timing settings and distinct seeds.")
    if any(not 0 <= s < 2**63 for s in args.seeds) or (args.verify_only and args.resume):
        parser.error("Use seeds in [0, 2**63) and resume only native runs.")
    torch.set_num_threads(1)
    arms = crossover_arms(args)
    if args.verify_only:
        result = verify(arms)
        args.out.mkdir(parents=True, exist_ok=False)
        (args.out/"verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result))
        return
    if os.name != "nt":
        parser.error("The native adapter requires Windows.")
    spec = credit.run_specification(args, arms)
    spec["controller"]["kind"] = "pure_frozen_ppo_bomb_crossover"
    spec["runtime"]["source_sha256"][__name__] = digest(Path(__file__))
    if not args.resume:
        args.out.mkdir(parents=True, exist_ok=False)
    with output_lock(args.out):
        report = credit.run_evaluation(args, arms, args.out/"report.json", spec)
    print(json.dumps(dict(matched_blocks=report["matched_blocks"], summary=report["matched_summary"])))


if __name__ == "__main__":
    main()
