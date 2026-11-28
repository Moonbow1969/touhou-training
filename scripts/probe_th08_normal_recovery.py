"""Compare TH08 Stage 1 policy cameras near observed life losses to human replay cameras."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.checkpoint_storage import digest
from scripts.qualify_th08_trial_credit import changes
from scripts.th08_camera_history import reconstruct
from scripts.train_five_original_games import freeze
from scripts.train_human_camera_policy import load_replay


TITLE_EXE = "3a6b4a0de24f916e669cf642ffd4f7c22ada1d319d9ce8ca8494fe96a2f5f5c1"
K = 8
BANK_PER_PILOT = 384
BEFORE_LOSS = 8
LOCAL_REFERENCE = 32


def replays(data, manifest):
    choices = json.loads(manifest.read_bytes())["qualified"]
    for row in choices:
        if row["stage"] != 1 or row["difficulty"] != "Normal":
            continue
        path = data / row["path"]
        if row["team"] != "Ghost Team" or row["original_executable_sha256"] != TITLE_EXE or digest(path) != row["archive_sha256"]:
            raise ValueError(f"Unmatched original human replay {path}")
        replay = load_replay(path)
        yield row["player_group"], replay["raw"]


def originals(root, endpoints):
    for receipt in endpoints["credits"]:
        if receipt.get("verified_stage") != 1 or receipt.get("role") != "held":
            continue
        receipt_path = root / "receipts" / f"{receipt['arm']}-{receipt['seed']}.json"
        actual = json.loads(receipt_path.read_bytes())
        if digest(receipt_path) != receipt["signed_receipt_sha256"] or any(receipt[k] != v for k, v in actual.items()):
            raise ValueError(f"Changed original receipt for registered endpoint {receipt_path}")
        if (receipt["rank"] != "Normal" or receipt["team"] != "Ghost" or
            receipt["source_original"]["th08tr.exe"] != TITLE_EXE or receipt["actions"]["outcome"] != "game_over"):
            raise ValueError(f"Unmatched original policy credit {receipt_path}")
        folder = root / receipt["source"]
        behavior = folder / "behavior.npz"
        timing = folder / "timing.jsonl"
        raw = json.loads((folder / "receipt.json").read_bytes())
        if digest(behavior) != receipt["behavior_sha256"] or digest(timing) != raw["timing_sha256"]:
            raise ValueError(f"Changed original trace {receipt_path}")
        observations = [json.loads(line) for line in timing.read_bytes().splitlines()]
        playable = [r for r in observations if r["phase"] == "policy"]
        if observations[-1]["phase"] != "terminal" or len(playable) != receipt["native_decisions"]:
            raise ValueError("Inconsistent original physical stop")
        readings = [r["spares"] for r in playable + observations[-1:]]
        # Stop at non-terminal spare loss evidence. An animated Continue prompt is a later visual state.
        events = [i for i, delta in changes(readings, 2) if delta < 0 and i >= LOCAL_REFERENCE]
        with np.load(behavior, allow_pickle=False) as archive:
            frames = archive["frames"]
        yield receipt, reconstruct(frames, 0, len(frames), 1), events


@torch.inference_mode()
def embeddings(module, camera, motion, device):
    part = []
    for block in torch.from_numpy(camera).split(64):
        block = block.to(device).float() / 255
        if not motion:
            block[:, 1] = 0
        part.append(module(block))
    return torch.cat(part)


def distances(features, bank):
    bank = torch.nn.functional.normalize(bank, dim=1)
    out = []
    for chunk in features.split(256):
        direction = torch.nn.functional.normalize(chunk, dim=1)
        out.append((1 - (direction @ bank.T).topk(K, dim=1).values).mean(1).cpu())
    return torch.cat(out).numpy()


def measure(module, humans, held, native, motion, device):
    bank = torch.cat([embeddings(module, data, motion, device) for _, data in humans])
    mean = bank.mean(0)
    bank -= mean
    reference = np.sort(np.concatenate([distances(embeddings(module, data, motion, device) - mean, bank)
                                        for _, data in held]))
    episodes = []
    for credit, cameras, events in native:
        measured = distances(embeddings(module, cameras, motion, device) - mean, bank)
        percentile = (np.searchsorted(reference, measured) * 100 / len(reference))
        windows = []
        for event in events:
            pre = percentile[event-BEFORE_LOSS:event]
            baseline = percentile[event-LOCAL_REFERENCE:event-2*BEFORE_LOSS]
            windows.append(dict(verified_spare_loss_action=event, last_8_median_percentile=float(np.median(pre)),
                                preceding_16_median_percentile=float(np.median(baseline)),
                                change_percentile=float(np.median(pre) - np.median(baseline))))
        episodes.append(dict(seed=credit["seed"], arm=credit["arm"], decisions=len(cameras),
                             full_median_percentile=float(np.median(percentile)), deaths=windows))
    return dict(human_bank_frames=len(bank), human_percentile_reference_frames=len(reference), credits=episodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--originals", type=Path, required=True)
    parser.add_argument("--endpoints", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source = args.data / "th08-trial002-human"
    parent = args.data / "th08-normal-temporal/change-free/policy.ts"
    module = torch.jit.load(str(parent), map_location=device).eval().features
    train = list(replays(args.data, source / "train-qualified-33-stages-frozen.json"))
    held = list(replays(args.data, source / "held-qualified-12-stages-frozen.json"))
    if len(train) != 5 or len(held) != 2 or {r[0] for r in train} & {r[0] for r in held}:
        raise ValueError("Stage 1 original player split changed")
    bank = [(group, frames[np.linspace(0, len(frames)-1, BANK_PER_PILOT, dtype=np.int64)]) for group, frames in train]
    native = list(originals(args.originals, json.loads(args.endpoints.read_bytes())))
    if not native:
        raise ValueError("No qualified Stage 1 original policy credits")
    freeze(args.out, dict(prior=digest(parent), endpoint_evidence_sha256=digest(args.endpoints), k=K, checkpoint_device=device,
           restriction=("Statistics are descriptive. Both camera encodings use the exact same original build and Stage 1. "
                        "Replay and policy cadence and rendering can differ. Gray-only removes the time-difference channel but does not prove matching sensors. "
                        "An original stock change follows the collision on screen and cannot identify the last safe action."),
           comparisons={"camera_pair": measure(module, bank, held, native, True, device),
                        "gray_only": measure(module, bank, held, native, False, device)}))


if __name__ == "__main__":
    main()
