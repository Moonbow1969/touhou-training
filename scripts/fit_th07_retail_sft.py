"""Fit three frozen-retail SFT policies with balanced author-stage exposure."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F

from fit_th07_locator_ablation import cell_id, evaluate, forward, sha, write
from th07_retail_bank import held_stages, training_stages


def tensor_data(stage):
    return tuple(torch.from_numpy(value) for value in
                 (stage.images, stage.actions, stage.xy, stage.changed_direction, stage.native_tick))


def join(strata):
    return tuple(torch.cat(field) for field in zip(*strata))


def compare_export(model, images, device):
    with torch.no_grad():
        x = images[:12].to(device).float() / 255
        runtime_heads = model(x)[:4]
        differentiable_heads, _ = forward(model, x)
        error = max((a-b).abs().max().item() for a, b in zip(runtime_heads, differentiable_heads))
        if error >= 3e-5:
            raise ValueError(f"Export mismatch {error}")
        return error


def fit(args, cfg, training, device):
    groups = sorted(training)
    train = join([training[key] for key in groups])
    n = cfg["required_windows_per_player_stage"]
    per_stratum = cfg["examples_per_stratum_per_batch"]
    assert n % per_stratum == 0 and cfg["batch"] == len(groups) * per_stratum
    offsets = (torch.arange(len(groups)) * n)[:, None]
    teacher = {}
    if "parent_kl_beta" in cfg:
        frozen = torch.jit.load(str(args.parent), map_location=device).eval()
        with torch.inference_mode():
            cached = {1: [], 3: []}
            for batch in train[0].split(128):
                logits = frozen(batch.to(device).float() / 255)
                for h in cached:
                    cached[h].append(logits[h].softmax(-1).cpu())
            teacher = {h: torch.cat(rows) for h, rows in cached.items()}
        del frozen
    for seed in cfg["seeds"]:
        folder = args.out / str(seed)
        folder.mkdir()
        torch.manual_seed(seed)
        rng = torch.Generator().manual_seed(seed)
        model = torch.jit.load(str(args.parent), map_location=device).eval()
        parity = compare_export(model, train[0], device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        start = time.monotonic()
        epochs = []
        for epoch in range(1, cfg["epochs"]+1):
            perm = torch.stack([torch.randperm(n, generator=rng) for _ in groups]) + offsets
            batches = perm.reshape(len(groups), -1, per_stratum).transpose(0, 1).reshape(-1, cfg["batch"])
            assert torch.equal(batches.flatten().sort().values, torch.arange(len(train[0])))
            assert torch.all(torch.bincount((batches[0] // n), minlength=len(groups)) == per_stratum)
            epoch_losses = []
            for rows in batches:
                images = train[0][rows].to(device).float() / 255
                targets = train[1][rows].to(device)
                actions, locator = forward(model, images)
                head_losses = [F.cross_entropy(logit, targets[:, h]) for h, logit in enumerate(actions)]
                xy_loss = F.cross_entropy(locator, cell_id(train[2][rows].to(device)))
                opt.zero_grad(set_to_none=True)
                if teacher:
                    kl = [F.kl_div(actions[h].log_softmax(-1), teacher[h][rows].to(device),
                                   reduction="batchmean") for h in (1, 3)]
                    action_objective = head_losses[0] + head_losses[2] + cfg["parent_kl_beta"] * sum(kl)
                else:
                    kl = []
                    action_objective = sum(head_losses)
                (action_objective + cfg["xy_weight"] * xy_loss).backward()
                opt.step()
                epoch_losses.append(torch.stack([v.detach() for v in (*head_losses, xy_loss, *kl)]))
            losses = torch.stack(epoch_losses).mean(0).cpu().tolist()
            stats = dict(epoch=epoch, optimizer_steps=len(batches), train_examples=batches.numel(),
                         action_ce=losses[:4], spatial_ce=losses[4], wall_seconds=time.monotonic()-start,
                         ordered_rows_sha256=hashlib.sha256(batches.numpy().astype("<i8").tobytes()).hexdigest())
            if teacher:
                stats["kl_focus_bomb"] = losses[5:]
            epochs.append(stats)
            print("EPOCH", seed, json.dumps(stats), flush=True)
        export_parity = compare_export(model, train[0], device)
        model.to("cpu").save(str(folder / "policy.ts"))
        write(folder / "fit.json", dict(seed=seed, epochs=epochs, parent_sha256=sha(args.parent),
                                       policy_sha256=sha(folder / "policy.ts"), initial_forward_error=parity,
                                       final_forward_error=export_parity))
        del model, opt


def score(args, cfg, device):
    held = {}
    for stage in held_stages(args.study, args.source, args.index_sha256):
        assert len(stage.actions) == cfg["required_windows_per_player_stage"]
        held[stage.source_id, stage.stage] = tensor_data(stage)
    assert set(held) == {(sid, s) for sid in cfg["source_ids"]["held"] for s in cfg["stages"]}
    outputs = {}
    for label, path in [("parent", args.parent)] + [(str(s), args.out / str(s) / "policy.ts") for s in cfg["seeds"]]:
        if label != "parent":
            assert sha(path) == json.loads((path.parent / "fit.json").read_bytes())["policy_sha256"]
        model = torch.jit.load(str(path), map_location=device).eval()
        audits, all_errors = {}, []
        for (sid, st), data in held.items():
            metrics, errors = evaluate(model, data, torch.arange(len(data[0])), device)
            audits[f"{sid}-s{st}"] = metrics
            all_errors.append(errors)
        author_metrics = {}
        for sid in cfg["source_ids"]["held"]:
            data = join(held[sid, st] for st in cfg["stages"])
            metrics, _ = evaluate(model, data, torch.arange(len(data[0])), device)
            author_metrics[str(sid)] = metrics
        outputs[label] = dict(sha256=sha(path), stages=audits, authors=author_metrics,
                              move=float(np.mean([a["accuracy"][0] for a in author_metrics.values()])),
                              switch=float(np.mean([a["switch_move_acc"] for a in author_metrics.values()])),
                              locator_median=float(np.median(np.concatenate(all_errors))))
    parent = outputs["parent"]
    children = [outputs[str(s)] for s in cfg["seeds"]]
    pooled = dict(delta_move=float(np.mean([m["move"]-parent["move"] for m in children])),
                  delta_switch=float(np.mean([m["switch"]-parent["switch"] for m in children])),
                  localization=float(np.mean([m["locator_median"] for m in children])))
    thresholds = cfg["gate"]
    passed = (pooled["delta_move"] >= thresholds["mean_delta_move_accuracy_min"] and
              pooled["delta_switch"] >= thresholds["mean_delta_switch_move_accuracy_min"] and
              pooled["localization"] <= thresholds.get("mean_seed_median_xy_error_max", float("inf")))
    write(args.out / "result.json", dict(models=outputs, mean=pooled, passed=passed,
                                         index_sha256=args.index_sha256, protocol_sha256=sha(args.protocol),
                                         gpu_peak_allocated_mb=torch.cuda.max_memory_allocated()/1e6 if device == "cuda" else 0))
    print("SEALED", json.dumps(pooled), passed, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "source", "protocol", "parent", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--index-sha256", required=True)
    args = parser.parse_args()
    cfg = json.loads(args.protocol.read_bytes())
    assert sha(args.parent) == cfg["parent_sha256"]
    assert sha(args.study / "protocol.json") == cfg["acquisition_protocol_sha256"]
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    training = {}
    for stage in training_stages(args.study, args.source, args.index_sha256):
        assert len(stage.actions) == cfg["required_windows_per_player_stage"]
        training[stage.source_id, stage.stage] = tensor_data(stage)
    assert set(training) == {(sid, s) for sid in cfg["source_ids"]["train"] for s in cfg["stages"]}
    metadata = dict(train_examples=sum(len(t[0]) for t in training.values()), index_sha256=args.index_sha256,
                    initial_checkpoint=sha(args.parent), protocol=sha(args.protocol), seeds=cfg["seeds"],
                    pid=os.getpid(), device=device, torch_version=torch.__version__,
                    hardware=(torch.cuda.get_device_name() if device == "cuda" else "CPU"),
                    implementation={f.name: sha(f) for f in (Path(__file__), Path(__file__).with_name("th07_retail_bank.py"),
                                                               Path(__file__).with_name("fit_th07_locator_ablation.py"))})
    write(args.out / "bindings.json", metadata)
    print("START", json.dumps(metadata), flush=True)
    fit(args, cfg, training, device)
    del training
    score(args, cfg, device)


if __name__ == "__main__":
    main()
