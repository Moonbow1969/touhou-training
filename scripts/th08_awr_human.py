"""Encode the frozen TH08 Normal human replay split for matched AWR."""

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.checkpoint_storage import digest
from scripts.train_human_camera_policy import load_replay
from scripts.train_th08_normal_temporal import history_windows, read


SPLITS = {
    "train": (33, "c066e647508681556e366a3cb3a50b05047cbcf024f175d79fa2d7f35683a8fd", 5, 58696),
    "held": (12, "2ae099d26bb8418808339ba287367184aa04ac01cbdab3494a200595078aecaf", 2, 15558),
}
SOURCE_FIELDS = (
    "title", "game_version", "difficulty", "team", "player_group", "split", "stage",
    "replay_sha256", "source_sha256", "native_input_qualification_sha256",
    "original_executable_sha256",
)


def load_human(data: Path, encoder: nn.Module, device: torch.device) -> dict:
    """Return per-split CPU (frozen features, four actions, movement-switch mask)."""
    root = data / "th08-trial002-human"
    tables = {}
    groups = {}
    for role, (number, sha, authors, _) in SPLITS.items():
        manifest = root / f"{role}-qualified-{number:02}-stages-frozen.json"
        if digest(manifest) != sha:
            raise ValueError(f"Changed frozen TH08 human {role} manifest: {manifest.name}")
        table = read(manifest)
        rows = [row for row in table["qualified"] if row["difficulty"] == "Normal"]
        groups[role] = {row["player_group"].casefold() for row in rows}
        if (table["split"] != role or len(groups[role]) != authors
                or any((row["title"], row["game_version"], row["team"], row["split"])
                       != ("TH08", "0.02_", "Ghost Team", role) for row in rows)):
            raise ValueError(f"Incorrect author or original-build registration in human {role}")
        tables[role] = rows
    if groups["train"] & groups["held"]:
        raise ValueError("The human AWR training and held author groups overlap")
    rows = [row for table in tables.values() for row in table]
    if len({row["archive_sha256"] for row in rows}) != len(rows):
        raise ValueError("A human archive is repeated in the frozen TH08 cohort")
    if ({r["replay_sha256"] for r in tables["train"]}
            & {r["replay_sha256"] for r in tables["held"]}):
        raise ValueError("The same source replay is assigned to train and held")

    result = {}
    sources = {}
    encoder.eval()
    with torch.no_grad():
        for role, rows in tables.items():
            features, actions, switches = [], [], []
            author_counts = Counter()
            stage_sources = []
            for row in rows:
                path = data / row["path"]
                metadata = read(path.with_suffix(".json"))
                if any(metadata.get(key) != row[key] for key in SOURCE_FIELDS):
                    raise ValueError(f"Changed original camera registration: {path.name}")
                replay = load_replay(path)
                if replay["sha256"] != row["archive_sha256"]:
                    raise ValueError(f"Changed original camera archive: {path.name}")
                ticks = replay["native_frame"]
                if np.any(np.diff(ticks) <= 0):
                    raise ValueError(f"Non-increasing original replay clock: {path.name}")
                # The temporal experiment qualified these windows. AWR retains
                # its latest causal two-channel camera and the same targets.
                current = history_windows(ticks)[:, -1]
                labels = replay["action"][current]
                if not len(current) or np.any(labels[:, 1]):
                    raise ValueError(f"Unexpected TH08 Normal action support: {path.name}")
                for offset in range(0, len(current), 128):
                    image = replay["raw"][current[offset:offset + 128]]
                    feature = encoder(torch.from_numpy(image).to(device).float().div_(255))
                    if feature.shape != (len(image), 128):
                        raise ValueError("TH08 AWR requires a 128-dimensional frozen encoder")
                    features.append(feature.detach().cpu())
                actions.append(torch.from_numpy(labels))
                switches.append(torch.from_numpy(labels[:, 0] != replay["action"][current - 1, 0]))
                author_counts[row["player_group"]] += len(current)
                stage_sources.append(dict(stage=row["stage"], author=row["player_group"],
                                          archive_sha256=replay["sha256"], decisions=len(current)))
                print(f"Verified human {role} {row['player_group']} S{row['stage']}: {len(current)}", flush=True)
            copied = (torch.cat(features), torch.cat(actions), torch.cat(switches))
            _, sha, _, decisions = SPLITS[role]
            if len(copied[0]) != decisions or not torch.isfinite(copied[0]).all():
                raise ValueError(f"Unexpected human camera count or feature values in {role}")
            result[role] = copied
            sources[role] = dict(manifest_sha256=sha, decisions=decisions,
                                 decisions_by_author=dict(author_counts), original_stages=stage_sources)
    result["_sources"] = sources
    return result
