"""Compare two frozen-feature Stage 1 movement heads against their exact parent."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "work/th14-rl").is_dir():
    ROOT = Path.cwd()
REPO = ROOT / "work/th14-rl"
sys.path.insert(0, str(REPO))
from scripts.evaluate_stage_policies import read_policy
from scripts.pretrain_policy import movement_direction, read_shard

STORE = ROOT / "outputs/touhou-sft-world"
DATA = STORE / "sft-native-data-261123/frozen-training-snapshot"
PARENT = STORE / "overnight-title-equal-61/sft/policy.pt"
OUT = ROOT / "outputs/th14-stage1-movement-head-261125"
TRAIN = ("th14_ud0864", "th14_ud05e0", "th14_ud0613")
HOLDOUT = ("th14_ud06d6", "th14_ud068f")
STEPS, RATE, KL = 100, 0.0008, 0.75


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@torch.no_grad()
def load_replay(replay, parent):
    path = DATA / f"{replay}-stage1.npz"
    info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert (info["stage"], info["difficulty"], info["shot"].replace(" ", "")) == (1, "Normal", "ReimuA")
    assert info["replay"] == replay + ".rpy"
    metadata = dict(info, samples=info["pairs"])
    bullets, ctx, labels, soft = read_shard((path, metadata), motion_history=True, stage_clock=True, soft_movement=True)
    latents = []
    for start in range(0, len(ctx), 128):
        obs = {"bullets": torch.from_numpy(bullets[start:start+128]), "context": torch.from_numpy(ctx[start:start+128])}
        latents.append(parent.mlp_extractor.forward_actor(parent.extract_features(obs)))
    with np.load(path, allow_pickle=False) as data:
        previous = movement_direction(data["native"])
        first = movement_direction(data["expert_action_chunk"][:, 0])
    return dict(x=torch.cat(latents), soft=torch.tensor(soft), first=torch.tensor(first), previous=torch.tensor(previous),
                modal=torch.tensor(labels[:, 0]), metadata=dict(replay=replay, player=info["player"], source_split=info["split"],
                pairs=info["pairs"], npz_sha256=digest(path), replay_sha256=info["sha256"]))


def cases(first, previous):
    return {"starts": (previous == 4) & (first != 4), "turns": (previous != 4) & (first != 4) & (first != previous),
            "idle_holds": (previous == 4) & (first == 4), "active_holds": (previous != 4) & (first == previous),
            "stops": (previous != 4) & (first == 4)}


def fit(rows, parent, event_weight):
    x, first, previous = (torch.cat([r[name] for r in rows]) for name in ("x", "first", "previous"))
    flags = cases(first, previous)
    weights = 1 + (event_weight - 1) * (flags["starts"] | flags["turns"]).float()
    head = torch.nn.Linear(x.shape[1], 9)
    with torch.no_grad():
        head.weight.copy_(parent.action_net.weight[:9])
        head.bias.copy_(parent.action_net.bias[:9])
        target = head(x).softmax(1)
    optimizer = torch.optim.Adam(head.parameters(), lr=RATE)
    for _ in range(STEPS):
        logits = head(x)
        imitation = (weights * F.cross_entropy(logits, first, reduction="none")).sum() / weights.sum()
        anchor = F.kl_div(F.log_softmax(logits, dim=1), target, reduction="batchmean")
        optimizer.zero_grad()
        (imitation + KL * anchor).backward()
        optimizer.step()
    return head


@torch.no_grad()
def score(row, head):
    logits = head(row["x"])[:, :9]
    first, previous = row["first"], row["previous"]
    guess = logits.argmax(1)
    slices = {}
    for name, flag in cases(first, previous).items():
        n = int(flag.sum())
        hits = int((guess[flag] == first[flag]).sum())
        slices[name] = dict(n=n, correct=hits, accuracy=hits/n if n else None)
    idle = slices["idle_holds"]
    return dict(slices=slices, idle_false_move_rate=1-idle["accuracy"],
                first_tick_nll=float(F.cross_entropy(logits, first)),
                four_tick_nll=float(-(row["soft"] * logits.log_softmax(1)).sum(1).mean()),
                modal_accuracy=float((guess == row["modal"]).float().mean()))


def main():
    torch.manual_seed(211)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert digest(PARENT) == "d62aea41897019df9bec90f3f3288a350a0dfbd1bd471fbee0377400cdc1e2bc"
    parent = read_policy(PARENT, "cpu", motion_history=True)
    train = [load_replay(name, parent) for name in TRAIN]
    heads = {"parent":parent.action_net, "uniform":fit(train, parent, 1), "event4":fit(train, parent, 4)}
    hold = [load_replay(name, parent) for name in HOLDOUT]
    training_players = {r["metadata"]["player"].casefold() for r in train}
    assert all(r["metadata"]["player"].casefold() not in training_players for r in hold)
    report = {"parent_sha256":digest(PARENT), "runner_sha256":digest(Path(__file__)),
              "config":{"objective":"immediate next-tick direction CE + KL(parent || head); event4 gives starts and active turns 4x CE",
                        "trainable":"first 9 weights/bias rows of final actor layer; frozen pretrained actor features",
                        "optimizer":"full-batch Adam", "steps":STEPS, "learning_rate":RATE,"parent_kl":KL,
                        "device":"CPU", "threads":2,"torch":torch.__version__, "seed":211, "checkpoint_selection":"final fixed update"},
              "candidate_gate":"each held-out replay: +2 percentage points or more in BOTH start and turn exact direction vs parent; idle false-move increase <=2pp and four-tick NLL increase <=0.02",
              "holdout_scope":"Toshi is repeatedly used project development data; Aria is in an expanded project's training pool, excluded by whole replay from this experiment and the historical parent. Sealed test unopened.",
              "data":[r["metadata"] for r in train+hold],
              "roles":{"train": TRAIN, "held_out": HOLDOUT},
              "train_event_counts":{kind:int(sum(cases(r["first"],r["previous"])[kind].sum().item() for r in train)) for kind in ("starts","turns")},
              "held_out":{}}
    for row in hold:
        results = {name:score(row, head) for name, head in heads.items()}
        report["held_out"][row["metadata"]["replay"]] = results
    def passes(table):
        p, e = table["parent"], table["event4"]
        return (all(e["slices"][k]["accuracy"] >= p["slices"][k]["accuracy"] + .02 for k in ("starts","turns"))
                and e["idle_false_move_rate"] <= p["idle_false_move_rate"] + .02 and e["four_tick_nll"] <= p["four_tick_nll"] + .02)
    report["promote_to_native_screen"] = all(passes(t) for t in report["held_out"].values())
    OUT.mkdir(exist_ok=True)
    with torch.no_grad():
        parent.action_net.weight[:9].copy_(heads["event4"].weight)
        parent.action_net.bias[:9].copy_(heads["event4"].bias)
    torch.save(parent.state_dict(), OUT / "policy.pt")
    # Sidecar carries only inference configuration. Parent metrics do not describe this candidate.
    info = json.loads(PARENT.with_name("metrics.json").read_text(encoding="utf-8"))
    keys = ("hidden","encoder","stage_clock","motion_history","retain_recent_motion","motion_spans","factorized_movement")
    (OUT/"metrics.json").write_text(json.dumps({k:info[k] for k in keys if k in info}, indent=2),encoding="utf-8")
    check = read_policy(OUT/"policy.pt", "cpu", motion_history=True)
    for row in hold:
        assert score(row, check.action_net) == report["held_out"][row["metadata"]["replay"]]["event4"]
    report["candidate_sha256"] = digest(OUT/"policy.pt")
    (OUT/"comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False),encoding="utf-8")
    if Path(__file__).resolve() != (OUT/Path(__file__).name).resolve():
        shutil.copyfile(__file__, OUT/Path(__file__).name)
    print(json.dumps({"promote":report["promote_to_native_screen"], "weights":report["candidate_sha256"],"output":str(OUT)}))


if __name__ == "__main__":
    main()
