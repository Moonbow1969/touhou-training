"""Fit a bounded TH08 Normal online PPO update to the current-camera SFT actor."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze
from scripts.train_full_resolution_camera_ppo import chosen, credit, logits_on
from scripts.train_replay_camera_encoder import CameraPolicy
from scripts.train_th08_normal_temporal import Actor, Readout, WIDTHS


TRAIN_SEEDS = {80960101, 80960102, 80960103, 80960104}
EXPECTED_PARENT = "a75ce901423c76bb2db1d58a46cda7a06d0c2474d0c234eef41093e00140c1b5"
ORIGINAL_EXE = "3a6b4a0de24f916e669cf642ffd4f7c22ada1d319d9ce8ca8494fe96a2f5f5c1"
EPOCHS = 2
BATCH = 96
MICROBATCH = 32
RATE = 1.5e-5
KL_CAP = .02


class CurrentActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = Actor(CameraPolicy(), Readout(), False)
        self.critic = nn.Linear(128, 1)
        nn.init.zeros_(self.critic.weight)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x):
        current = self.actor.features(x)
        history = current.unsqueeze(1).expand(-1, 4, -1)
        prior = self.actor.prior(current).split(WIDTHS, 1)
        return (*self.actor.readout(history, prior), self.critic(current))


def source_rows(manifest, expected_parent):
    rows = json.loads(manifest.read_bytes())
    if len(rows) != len(TRAIN_SEEDS) or {row["seed"] for row in rows} != TRAIN_SEEDS:
        raise ValueError("Expected four complete registered fresh Normal TRAIN games")
    for row in sorted(rows, key=lambda item: item["seed"]):
        receipt_path = manifest.parent / "native08/receipts" / f"single-state-{row['seed']}.json"
        receipt_hash = digest(receipt_path)
        if row["receipt_sha256"] != receipt_hash:
            raise ValueError(f"Original receipt hash mismatch for seed {row['seed']}")
        receipt = json.loads(receipt_path.read_bytes())
        audit_path = receipt_path.parent.parent / "terminal-audits" / f"{row['seed']}.json"
        audit = json.loads(audit_path.read_bytes())
        if (row["title"] != 8 or row["difficulty"] != 1 or receipt["seed"] != row["seed"] or
                receipt["rank"] != "Normal" or receipt["team"] != "Ghost" or receipt["role"] != "train" or
                receipt["history_states"] != 1 or receipt["policy_sha256"] != expected_parent or
                receipt["actions"]["outcome"] != "game_over" or audit["zero_continue"] is not True or
                audit["signed_arm_receipt_sha256"] != receipt_hash or audit["optimizer_sha256"] != row["sha256"] or
                audit["seed"] != row["seed"] or audit["bootstrap"] != 0 or not audit["final_logged_reward_is_terminal"] or
                audit["rank_verified_original_samples"] != receipt["native_decisions"] or
                receipt["source_original"]["th08tr.exe"] != ORIGINAL_EXE or
                receipt["first_stock"] != [2, 3] or receipt["final_stock"][0] != 0 or
                audit["observed_stage"] != receipt["terminal_stage"] or receipt["terminal_stage"] not in (None, 1, 2, 3)):
            raise ValueError(f"Unqualified original credit for seed {row['seed']}")
        yield row, receipt


def optimize(model, packed, device, *, kl_cap=KL_CAP):
    observations, actions, old_logp, advantages, returns, old_heads = packed
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-7)
    optimizer = torch.optim.Adam(model.parameters(), lr=RATE, eps=1e-5)
    epochs = []
    for epoch in range(1, EPOCHS + 1):
        previous = {key: value.detach().clone() for key, value in model.state_dict().items()}
        stats = []
        permutation = torch.randperm(len(observations))
        for ids in permutation.split(BATCH):
            optimizer.zero_grad(set_to_none=True)
            mean_loss = 0.
            for chunk in ids.split(MICROBATCH):
                *heads, value = model(observations[chunk].to(device).float() / 255)
                action = actions[chunk].to(device)
                ratio = (chosen(heads, action).sum(1) - old_logp[chunk].to(device)).exp()
                adv = advantages[chunk].to(device)
                actor = -torch.minimum(ratio * adv, ratio.clamp(.9, 1.1) * adv).mean()
                critic = F.smooth_l1_loss(value[:, 0], returns[chunk].to(device))
                entropy = sum((head.softmax(1) * head.log_softmax(1)).sum(1).mean() for head in heads)
                objective = (actor + .25 * critic + .002 * entropy) * (len(chunk) / len(ids))
                objective.backward()
                mean_loss += float(objective.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
            optimizer.step()
            stats.append(mean_loss)
        *current, predicted_values = logits_on(model, observations, device)
        kl = sum(float(F.kl_div(new.log_softmax(1), old.softmax(1), reduction="batchmean"))
                 for new, old in zip(current, old_heads.split(WIDTHS, 1)))
        accepted = -1e-7 <= kl <= kl_cap
        row = dict(epoch=epoch, joint_kl=kl, optimizer_steps=len(stats), accepted=accepted,
                   mean_loss=sum(stats) / len(stats), critic_rmse=float((predicted_values[:, 0] - returns).square().mean().sqrt()))
        epochs.append(row)
        print(row, flush=True)
        if not accepted:
            model.load_state_dict(previous)
            break
    if not any(row["accepted"] for row in epochs):
        raise ValueError("All attempted PPO epochs exceeded the registered KL gate")
    return epochs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(809601)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parent = digest(args.parent)
    if parent != EXPECTED_PARENT:
        raise ValueError("TH08 rollout parent policy changed")
    model = CurrentActorCritic().to(device)
    original = torch.jit.load(str(args.parent), map_location=device).eval()
    model.actor.load_state_dict(original.state_dict())
    training, credits = [], []
    for row, receipt in source_rows(args.manifest, parent):
        tensors, evidence = credit(row, args.manifest, model, parent, device)
        if evidence["steps"] != receipt["native_decisions"] or evidence["max_native_value_error"] != 0.:
            raise ValueError(f"Original action count or zero-critic mismatch in seed {row['seed']}")
        training.append(tensors)
        credits.append(dict(evidence, trace=row["sha256"], terminal_stage=receipt["terminal_stage"],
                            first_stock=receipt["first_stock"], final_stock=receipt["final_stock"]))
    packed = tuple(torch.cat(tensors) for tensors in zip(*training))
    args.out.mkdir(parents=True, exist_ok=False)
    freeze(args.out / "protocol.json", dict(parent=parent, behavior_manifest=digest(args.manifest),
           train_seeds=sorted(TRAIN_SEEDS), epochs=EPOCHS, batch=BATCH, learn_rate=RATE,
           joint_kl_cap=KL_CAP, gamma=.999, gae_lambda=.985, clip=.1, seed=809601,
           learner_sha256=digest(Path(__file__)),
           objective="Full direction / Focus / Shot / Bomb joint PPO from zero critic, using the qualified worker's resource and terminal reward."))
    epochs = optimize(model, packed, device)
    model = model.cpu().eval()
    example = packed[0][:4].float() / 255
    path = args.out / "policy.ts"
    with torch.inference_mode():
        scripted = torch.jit.trace(model, example)
        scripted.save(str(path))
        reloaded = torch.jit.load(str(path), map_location="cpu")
        assert all(torch.equal(a, b) for a, b in zip(model(example), reloaded(example)))
    torch.save(model.state_dict(), args.out / "policy.pt")
    freeze(args.out / "report.json", dict(parent=parent, policy=digest(path), optimizer_epochs=epochs,
           device=device, actual_train_decisions=len(packed[0]), credits=credits,
           optimizer_steps=sum(row["optimizer_steps"] for row in epochs if row["accepted"])))


if __name__ == "__main__":
    main()
