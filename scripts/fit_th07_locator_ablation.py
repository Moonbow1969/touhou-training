"""Fit and freeze the six registered TH07 policies, then evaluate them in one held pass per model."""
import argparse
import gzip
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def directions(words):
    bits = [(words & bit != 0).astype(np.int64) for bit in (16, 32, 64, 128)]
    up, down, left, right = bits
    return 4 + 3 * (down - up) + right - left


def pack(path, role, protocol):
    audit_file = path / "source-audit.json"
    assert sha(audit_file) == protocol["source_audit_sha256"][role]
    audit = json.loads(audit_file.read_text())
    assert audit["original_replay_sha256"] == protocol["sources"][role]
    for name in ("gray.raw.gz", "positions.npz"):
        assert sha(path / name) == audit["artifacts_sha256"][name]
    with np.load(path / "positions.npz", allow_pickle=False) as a:
        ticks = a["native_tick"]
        j = a["causal_pair_rows"]
        native_xy = a["xy_native"][j]
        labels = a["action_next_fourheads"].astype(np.int64)
        words = a["stage1_replay_words"]
        assert np.array_equal(a["native_word"], words[ticks])
        assert np.array_equal(a["previous_native_word"], words[ticks-1])
    with gzip.open(path / "gray.raw.gz") as f:
        frames = np.frombuffer(f.read(), np.uint8).reshape(-1, 112, 96)
    t = ticks[j]
    assert len(frames) == len(ticks) and np.all(t-ticks[j-1] == 4)
    assert np.all(np.diff(t)>0) and np.isfinite(native_xy).all()
    assert np.array_equal(labels[:,0], directions(words[t+1]))
    change = labels[:,0] != directions(words[t-3])
    delta = np.abs(frames[j].astype(np.int16) - frames[j-1]).astype(np.uint8)
    images = np.stack((frames[j], delta), axis=1)
    return tuple(map(torch.from_numpy, (images, labels, native_xy, change, t)))


def forward(model, images):
    camera = model.camera
    # The original trace stores this unused optional head's parameters, without an exported forward.
    spatial = camera.spatial(images)
    layers = list(camera.policy.children())
    hidden = layers[1](layers[0](spatial))
    assert hidden.shape[2:] == (28, 24)
    # Native adaptive averaging to 7x6 partitions this tensor into disjoint exact 4x4 bins.
    # Express those same means explicitly so CUDA can differentiate them deterministically.
    hidden = hidden.reshape(len(images), hidden.size(1), 7, 4, 6, 4).mean(dim=(3,5))
    for layer in layers[3:]:
        hidden = layer(hidden)
    actions = hidden.split((9,2,2,2), dim=1)
    locator = F.conv2d(spatial, camera.player.weight, camera.player.bias)
    assert locator.shape[1:] == (1, 56, 48)
    return actions, locator.flatten(1)


def cell_id(xy):
    cells = (xy / 8).floor().long()
    return cells[:, 1].clamp(0, 55) * 48 + cells[:, 0].clamp(0, 47)


def evaluate(model, data, ids, device):
    pred, nll, position_errors, bomb_prob = [], [], [], []
    images, y, xy, changed, _ = data
    with torch.no_grad():
        for rows in ids.split(64):
            images_batch = images[rows].to(device).float()/255
            actions = model(images_batch)[:4]
            _, location = forward(model, images_batch)
            head = [h.cpu() for h in actions]
            pred.append(torch.stack([h.argmax(1) for h in head], 1))
            nll.append(torch.stack([F.cross_entropy(h, y[rows, k], reduction="none") for k,h in enumerate(head)], 1))
            bomb_prob.append(head[3].softmax(1)[:,1])
            cell = location.argmax(1).cpu()
            loc = 8 * torch.stack((cell % 48, cell // 48), 1) + 4
            position_errors.append((loc-xy[rows]).norm(dim=1))
    predicted, losses, error = torch.cat(pred), torch.cat(nll), torch.cat(position_errors)
    truth = y[ids]
    correct = predicted == truth
    is_switch = changed[ids]
    metrics = dict(n=len(ids), switches=int(is_switch.sum()), accuracy=correct.float().mean(0).tolist(),
                   nll=losses.mean(0).tolist(), joint_nll=losses.sum(1).mean().item(),
                   switch_move_acc=correct[is_switch,0].float().mean().item(),
                   stable_move_acc=correct[~is_switch,0].float().mean().item(),
                   switch_move_nll=losses[is_switch,0].mean().item(),
                   stable_move_nll=losses[~is_switch,0].mean().item(),
                   bomb_positive=int(truth[:,3].sum()), predicted_bomb_positive=int(predicted[:,3].sum()),
                   bomb_true_positive=int(((truth[:,3]==1)&(predicted[:,3]==1)).sum()),
                   mean_bomb_probability=torch.cat(bomb_prob).mean().item(),
                   focus_positive=int(truth[:,1].sum()), predicted_focus_positive=int(predicted[:,1].sum()),
                   locator_median_native_px=float(np.median(error.numpy())),
                   locator_p90_native_px=float(np.quantile(error.numpy(),.9)),
                   first_last_tick=[int(data[4][ids[0]]),int(data[4][ids[-1]])])
    return metrics, error.numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ("protocol", "data", "parent", "out"):
        ap.add_argument("--"+arg, type=Path, required=True)
    args = ap.parse_args()
    p = json.loads(args.protocol.read_text())
    assert sha(args.parent) == p["parent_sha256"]
    args.out.mkdir(parents=True, exist_ok=True)
    assert not any(args.out.glob("*/policy.ts")), "Use an empty run directory."
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = pack(args.data / "train", "train", p)
    cut = len(train[0])*4//5
    ticks = train[4]
    tr = torch.arange(cut)[ticks[:cut]+4 < ticks[cut]-4]
    va = torch.arange(cut, len(ticks))
    assert len(tr) and int(ticks[tr[-1]]+4) < int(ticks[va[0]]-4)
    bindings = dict(protocol_sha256=sha(args.protocol), trainer_sha256=sha(Path(__file__)),
                    parent_sha256=sha(args.parent), device=device, torch_version=torch.__version__,
                    train_rows=len(tr), sanity_rows=len(va), boundary_purged_rows=cut-len(tr),
                    optimize_ticks=[int(ticks[tr[0]]),int(ticks[tr[-1]])], sanity_ticks=[int(ticks[va[0]]),int(ticks[va[-1]])],
                    pid=os.getpid())
    write(args.out/"bindings.json", bindings)
    print("START", json.dumps(bindings), flush=True)
    for seed in p["seeds"]:
        for arm, weight in p["arms"].items():
            start = time.monotonic()
            torch.manual_seed(seed)
            model = torch.jit.load(str(args.parent), map_location=device).eval()
            match_input = train[0][tr[:12]].to(device).float()/255
            with torch.no_grad():
                parental = model(match_input)[:4]
                reconstructed, _ = forward(model, match_input)
                max_difference = max((a-b).abs().max().item() for a,b in zip(parental, reconstructed))
                assert max_difference < 0.00003, max_difference
            optimizer = torch.optim.Adam(model.parameters(), lr=p["lr"])
            batches = torch.Generator().manual_seed(seed)
            folder = args.out / f"{seed}-{arm}"
            folder.mkdir()
            order_hash = hashlib.sha256()
            epochs = []
            for epoch in range(p["fixed_epochs"]):
                epoch_actions, epoch_positions = [], []
                perm = tr[torch.randperm(len(tr), generator=batches)]
                order_hash.update(perm.numpy().astype("<i8").tobytes())
                for rows in perm.split(p["batch"]):
                    input_batch = train[0][rows].to(device).float()/255
                    target = train[1][rows].to(device)
                    actions, locations = forward(model, input_batch)
                    action_loss = sum(F.cross_entropy(h,target[:,k]) for k,h in enumerate(actions))
                    loc_loss = F.cross_entropy(locations, cell_id(train[2][rows].to(device)))
                    optimizer.zero_grad(set_to_none=True)
                    (action_loss+weight*loc_loss).backward()
                    optimizer.step()
                    epoch_actions.append(float(action_loss.detach()))
                    epoch_positions.append(float(loc_loss.detach()))
                epochs.append(dict(epoch=epoch+1, action_ce=float(np.mean(epoch_actions)), spatial_ce=float(np.mean(epoch_positions)),
                                   steps=len(epoch_actions), wall_seconds=time.monotonic()-start))
                print(seed, arm, json.dumps(epochs[-1]), flush=True)
            with torch.no_grad():
                deployed = model(match_input)[:4]
                reconstructed, _ = forward(model, match_input)
                final_difference = max((a-b).abs().max().item() for a,b in zip(deployed, reconstructed))
                assert final_difference < 0.00003, final_difference
            model.to("cpu").save(str(folder/"policy.ts"))
            stats = dict(arm=arm,seed=seed,weight=weight,initial_deploy_max_abs=max_difference,final_deploy_max_abs=final_difference,order_sha256=order_hash.hexdigest(),epochs=epochs,
                         policy_sha256=sha(folder/"policy.ts"))
            write(folder/"fit.json", stats)
            del model, optimizer
    # Freeze every arm before opening the held source, and do not change weights after this line.
    held = pack(args.data/"held", "held", p)
    he = torch.arange(len(held[0]))
    results, pooled, differences = {}, [], []
    for seed in p["seeds"]:
        results[seed] = {}
        for arm in p["arms"]:
            folder = args.out / f"{seed}-{arm}"
            model = torch.jit.load(str(folder/"policy.ts"),map_location=device).eval()
            sanity, _ = evaluate(model, train, va, device)
            h, errors = evaluate(model, held, he, device)
            if arm == "action-xy":
                pooled.append(errors)
            fit = json.loads((folder/"fit.json").read_text())
            assert sha(folder/"policy.ts") == fit["policy_sha256"]
            results[seed][arm] = dict(held=h,sanity=sanity,order_sha256=fit["order_sha256"],policy_sha256=fit["policy_sha256"])
        a, b = (results[seed][n] for n in ("action","action-xy"))
        assert a["order_sha256"] == b["order_sha256"]
        differences.append(dict(seed=seed, move=b["held"]["accuracy"][0]-a["held"]["accuracy"][0],
                                switch=b["held"]["switch_move_acc"]-a["held"]["switch_move_acc"],
                                stable=b["held"]["stable_move_acc"]-a["held"]["stable_move_acc"],
                                locator=b["held"]["locator_median_native_px"]))
    avg = {k:float(np.mean([r[k] for r in differences])) for k in ("move","switch","stable","locator")}
    seen = sum(r["move"]>0 for r in differences)
    g = p["gate"]
    passed = avg["move"]>=g["mean_delta_move_accuracy"] and seen>=g["positive_move_seed_pairs_min"] and avg["switch"]>=g["mean_delta_changed_expert_direction_accuracy_min"] and avg["locator"]<=g["locator_max_native_px"]
    write(args.out/"result.json", dict(**bindings, results=results, paired=differences, mean=avg, positive_seeds=seen,
                                     pooled_locator_median=float(np.median(np.concatenate(pooled))), passed=passed,
                                     gpu_peak_allocated_mb=torch.cuda.max_memory_allocated()/1e6 if device=="cuda" else 0))
    print("SEALED", json.dumps(avg), passed, flush=True)


if __name__ == "__main__":
    main()
