"""Compare a full-policy PPO checkpoint with its SFT parent on fixed replay observations."""

import argparse
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from scripts.checkpoint_storage import MANIFEST, MIB, digest
from scripts.evaluate_stage_policies import read_policy
from scripts.pretrain_policy import batch_block, index_split, predict_heads, read_shard


DEFAULT_PLAN = (Path(__file__).resolve().parents[1] / "results/2026-11-23-th14-full-policy"
                / "replay-capture-split.json")
METRICS = ("greedy_idle_fraction", "sampled_idle_probability", "greedy_movement_accuracy",
           "movement_entropy_nats", "moving_probability_over_half_fraction", "focus_probability",
           "shoot_probability", "bomb_probability", "movement_expert_probability")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def reserved_replays(plan, data, sft_metrics):
    reserved = {row["replay"]: row for row in plan["test"]}
    if not reserved:
        raise ValueError("The split plan must reserve at least one independent test replay.")
    groups = {row["player_group_sha256"] for row in reserved.values()}
    raw_hashes = {row["replay_sha256"] for row in reserved.values()}
    input_rows = [load_json(path) for path in sorted(data.glob("*.json"))]
    input_rows.extend(sft_metrics["training_replays"])
    input_rows.extend(sft_metrics["validation_replays"])
    for row in input_rows:
        group = row.get("player_group") or row["player"].casefold()
        group_sha256 = sha256(group.encode("utf-8")).hexdigest()
        if (row.get("name", row.get("replay")) in reserved or group_sha256 in groups
                or row.get("sha256") in raw_hashes):
            raise ValueError("A reserved test replay or player is present in the audit or SFT inputs.")
    return [dict(replay=name, replay_sha256=row["replay_sha256"]) for name, row in reserved.items()]


def ppo_actor(path, base, base_hash):
    fingerprint = digest(path)
    with ZipFile(path) as archive:
        if MANIFEST in archive.namelist():
            if archive.getinfo(MANIFEST).file_size > MIB:
                raise ValueError("PPO checkpoint manifest exceeds 1 MiB.")
            manifest = json.loads(archive.read(MANIFEST))
            parent = manifest["compatibility"]["base_sha256"]
            steps = manifest["steps"]
            lineage = "embedded_manifest"
        else:
            report = load_json(path.with_name("report.json"))
            if report["checkpoints"]["ppo_sha256"] != fingerprint:
                raise ValueError("PPO digest differs from its legacy training report.")
            parent = report["checkpoints"]["base_sha256"]
            steps = report.get("checkpoint_steps", report.get("trained_steps"))
            lineage = "legacy_training_report"
        if parent != base_hash or type(steps) is not int or steps < 1:
            raise ValueError("The PPO checkpoint needs valid lineage from the supplied SFT parent.")
        if archive.getinfo("policy.pth").file_size > 64 * MIB:
            raise ValueError("PPO policy tensors exceed the offline audit's 64 MiB limit.")
        # The SB3 data and optimizer members are irrelevant to actor inference.
        state = torch.load(BytesIO(archive.read("policy.pth")), map_location="cpu", weights_only=True)
    actor = deepcopy(base)
    actor.load_state_dict(state, strict=True)
    return actor.eval(), {"sha256": fingerprint, "steps": steps, "lineage": lineage}


@torch.inference_mode()
def evaluate(shards, actors, *, stride=1, last_frame=None):
    base = actors["sft"]
    totals = {name: np.zeros(len(METRICS), dtype=np.float64) for name in actors}
    count = 0
    expert_idle = 0
    for shard in shards:
        arrays = read_shard(shard, motion_history=True, motion_spans=base.motion_spans,
                            retain_recent_motion=base.retain_recent_motion, stage_clock=base.stage_clock)
        indices = np.arange(0, len(arrays[-1]), stride)
        if last_frame is not None:
            with np.load(shard[0]) as data:
                indices = indices[data["stage_frame"][indices] <= last_frame]
        for bullets, context, actions in batch_block(arrays, 256, "cpu", indices):
            count += len(actions)
            expert_idle += int(actions[:, 0].eq(4).sum())
            for name, actor in actors.items():
                logits = predict_heads(actor, bullets, context)
                move, focus, shoot, bomb = (head.softmax(1) for head in logits)
                greedy = move.argmax(1)
                totals[name] += [
                    greedy.eq(4).sum(),
                    move[:, 4].sum(),
                    greedy.eq(actions[:, 0]).sum(),
                    -(move * logits[0].log_softmax(1)).sum(),
                    (move[:, 4] < .5).sum(),
                    focus[:, 1].sum(),
                    shoot[:, 1].sum(),
                    bomb[:, 1].sum(),
                    move.gather(1, actions[:, :1]).sum(),
                ]
    if not count:
        raise ValueError("The selected replay slice has no observations.")
    return {
        "decisions": count,
        "stride": stride,
        "last_stage_frame": last_frame,
        "expert_idle_fraction": expert_idle / count,
        "actors": {name: dict(zip(METRICS, (values / count).tolist())) for name, values in totals.items()},
    }


def evidence(shards):
    return [
        {"replay": meta["replay"], "stage": meta["stage"], "pairs": meta["samples"],
         "metadata_sha256": digest(path.with_suffix(".json")), "arrays_sha256": digest(path)}
        for path, meta in shards
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="SFT policy.pt with its metrics.json sidecar.")
    parser.add_argument("--ppo", type=Path, required=True, help="Full-policy PPO ZIP archive.")
    parser.add_argument("--data", type=Path, required=True, help="Frozen aligned replay snapshot directory.")
    parser.add_argument("--split-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--out", type=Path, required=True, help="Fresh JSON report path.")
    args = parser.parse_args()
    torch.set_num_threads(1)
    metrics_path = args.base.with_name("metrics.json")
    metrics = load_json(metrics_path)
    reserved = reserved_replays(load_json(args.split_plan), args.data, metrics)
    base_hash = digest(args.base)
    base = read_policy(args.base, "cpu", motion_history=True)
    if base.factorized_movement:
        parser.error("This audit expects the full-policy runner's categorical movement head.")
    ppo, ppo_evidence = ppo_actor(args.ppo, base, base_hash)
    validation = [shard for shard in index_split(args.data, "validation")
                  if (shard[1]["title"], shard[1]["stage"], shard[1]["difficulty"],
                      shard[1]["shot"].replace(" ", "")) == (14, 1, "Normal", "ReimuA")]
    test = [shard for shard in index_split(args.data, "test") if shard[1]["title"] == 17]
    actors = {"sft": base, "ppo": ppo}
    report = {
        "version": 1,
        "checkpoints": {"sft": {"sha256": base_hash, "metrics_sha256": digest(metrics_path)}, "ppo": ppo_evidence},
        "split_plan_sha256": digest(args.split_plan),
        "reserved_test_replays_excluded": reserved,
        "sampler": "Exact categorical probabilities and argmax on recorded human replay observations",
        "th14_normal_reimu_a_stage1_validation": {
            "inputs": evidence(validation),
            "full": evaluate(validation, actors),
            "first_500_stage_frames": evaluate(validation, actors, last_frame=500),
        },
        "th17_test_players": {"inputs": evidence(test), "every_fourth_decision": evaluate(test, actors, stride=4)},
    }
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved offline policy-drift report to {args.out}", flush=True)


if __name__ == "__main__":
    main()
