"""Collect exact camera trajectories and compare three frozen TH10 actors on the same host."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from scripts.collect_th10_full_action import save
from scripts.summarize_th10_full_credit import read_arm, summarize, draw


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint(path, history=False):
    while True:
        if path.is_file():
            try:
                model = torch.jit.load(str(path), map_location="cpu").eval()
                with torch.inference_mode():
                    sample=torch.zeros(1,2,112,96)
                    outputs = model(sample,torch.tensor([4],dtype=torch.long)) if history else model(sample)
                if [tuple(x.shape) for x in outputs[:4]] != [(1, 9), (1, 2), (1, 2), (1, 2)]:
                    raise ValueError(f"Unexpected camera policy outputs at {path}")
                return model
            except RuntimeError:
                if time.time() - path.stat().st_mtime > 60:
                    raise
        time.sleep(4)


def collect(args, arm, model, seeds, destination, active, camera, history=False):
    destination.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", "-m", "scripts.collect_th10_full_action",
           "--workers", str(args.workers), "--model", str(model),
           "--arm", arm, "--out", str(destination), "--expected-sha", digest(model),
           "--concurrency", str(active), "--seeds", *map(str, seeds)]
    if camera:
        cmd.append("--camera")
    if getattr(args,'difficulty',1)!=1:cmd.extend(['--difficulty',str(args.difficulty)])
    if getattr(args,'stage_sensor',None):cmd.extend(['--stage-sensor',args.stage_sensor])
    if history:cmd.append("--camera-history")
    host=getattr(args,"host","momentum")
    if host!="fleet":cmd.extend(["--only-host",host])
    with (destination / "collector.log").open("ab") as log:
        for attempt in range(3):
            run = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], stdout=log, stderr=log)
            if run.returncode == 0:
                return
            if attempt < 2:
                time.sleep(6)
        raise RuntimeError(f"Collection failed for {destination}; inspect collector.log.")


@torch.inference_mode()
def verify(directory, model_path, model, seeds, role, write=True, host="momentum",history=False,difficulty=1):
    sha = digest(model_path)
    receipts, audits = [], []
    for seed in seeds:
        actual_host=("accelerator" if seed%2 else "momentum") if host=="fleet" else host
        folder = directory / f"{actual_host}-{seed}"
        r = json.loads((folder / "report.json").read_bytes())
        assert r["camera_policy"] and r["terminal"] and r["continues"] == 0
        assert r['source'].get('difficulty',1)==difficulty
        assert r["seed"] == seed and r["policy_sha256"] == sha
        assert r["host"] == actual_host and r["stage_entries"][0]["lives"] == 2
        assert digest(folder / "endpoint.png") == r["terminal_screenshot_sha256"]
        behavior, pixels = folder / "behavior.npz", folder / "original-pixels.npz"
        assert digest(behavior) == r["behavior_sha256"]
        with np.load(behavior) as b, np.load(pixels) as p:
            n = r["steps"]
            assert p["obs"].shape == (n+1,112,96) and p["obs"].dtype == np.uint8
            assert np.array_equal(p["monotonic_ns"][:-1], b["clock_ns"])
            assert b["terminated"][-1] and b["terminated"].sum() == 1
            frames = torch.from_numpy(p["obs"][:-1]).float().div_(255)
            diff = torch.zeros_like(frames)
            diff[1:] = (frames[1:] - frames[:-1]).abs()
            images = torch.stack([frames, diff], dim=1)
            ls, vs = [], []
            if history:
                previous=torch.from_numpy(b['previous_move'].copy()).long()
                assert previous[0]==4 and np.array_equal(previous[1:].numpy(), b['action'][:-1,0])
            for position in range(0,n,20):
                batch=images[position:position+20]
                o = model(batch,previous[position:position+20]) if history else model(batch)
                ls.append(torch.cat(o[:4], 1)); vs.append(o[4].reshape(-1) if len(o)==5 else torch.zeros(len(batch)))
            logits = torch.cat(ls)
            selected = torch.stack([head.log_softmax(-1).gather(1, torch.from_numpy(b["action"][:, i].copy()).long()[:, None])[:,0]
                                    for i, head in enumerate(logits.split([9,2,2,2], 1))], 1).numpy()
            elog = float(np.abs(logits.numpy()-b["camera_logits"]).max())
            elp = float(np.abs(selected-b["logp"]).max())
            ev = float(np.abs(torch.cat(vs).numpy()-b["value"]).max())
            assert max(elog, elp, ev) < 3e-4, (seed, elog, elp, ev)
        audits.append(dict(seed=seed, steps=n, max_logits_error=elog, max_logp_error=elp, max_value_error=ev))
        receipts.append(dict(title=10, host=r["host"], seed=seed, role=role, policy_sha256=sha,
            file=str(behavior), behavior=str(behavior), path=str(behavior), sha256=r["behavior_sha256"],
            pixels=str(pixels), pixel_file=str(pixels), pixel_sha256=digest(pixels), steps=n,
            native_outcome=r["native_outcome"], outcome="game_over" if r["game_over"] else "ending"))
        if difficulty!=1:
            label={0:'Easy',2:'Hard',3:'Lunatic'}[difficulty]
            with gzip.open(folder/'native.csv.gz','rt') as raw:
                native_difficulties=[int(row['difficulty']) for row in csv.DictReader(raw)]
            assert len(native_difficulties)==r['native_samples']>0 and set(native_difficulties)=={difficulty}
            menu_hash=digest(folder/(label.lower()+'-rank.png'))
            assert r.get('difficulty_menu_sha256',menu_hash)==menu_hash
            receipts[-1].update(difficulty=difficulty,difficulty_label=label,native_difficulty_samples=r['native_samples'],
                                menu_sha256=menu_hash)
    if write:
        save(directory / "on-policy-validation.json", audits)
        save(directory / ("ppo-receipts.json" if role=="train" else "held-receipts.json"), receipts)
    return receipts, audits


def run(args):
    args.root = args.root.resolve()
    args.workers = args.workers.resolve()
    args.ppo3 = args.ppo3.resolve()
    torch.set_num_threads(2)
    base = args.root / "camera-native10"
    base.mkdir(parents=True, exist_ok=True)
    train, held = list(range(752101,752109)), list(range(754001,754007))
    sft, ppo = args.root / "camera-sft/policy.ts", args.root / "camera-ppo/policy.ts"
    design = ("Original TH10 official trial 0.02a Normal, full life stock, no Continue, no assisted actions. "
              "Momentum, same six controller seeds per held arm; native game RNG uncontrolled. "
              "The first three seeds get portable PPO3 before human camera SFT, the last three get human camera SFT first. "
              "Direct camera PPO follows the SFT training credits and uses sensor-matched evaluation. "
              "TH10 spirit strikes spend 1.00 Power. Real-time play depends on host speed.")
    save(base / "protocol.json", dict(design=design, train=train, held=held, hold_ms=120,
        camera_inputs="uint8 gray 112x96 and absolute change versus previous decision, divided by 255; first difference zero",
        actions=["movement 9","focus 2","shoot 2","spirit strike 2"]))
    sft_model = checkpoint(sft)
    train_dir = base / "train-sft"
    collect(args, "camera-sft", sft, train, train_dir, 3, True)
    verify(train_dir, sft, sft_model, train, "train")
    for arm, path, block in (("ppo3", args.ppo3, held[:3]), ("camera_sft",sft,held[3:]),
                             ("camera_sft",sft,held[:3]), ("ppo3",args.ppo3,held[3:])):
        collect(args, arm, path, block, base / ("held-"+arm), 3, arm != "ppo3")
    verify(base / "held-camera_sft", sft, sft_model, held, "held")
    portable_receipts = []
    for seed in held:
        folder = base / "held-ppo3" / f"momentum-{seed}"
        r = json.loads((folder/"report.json").read_bytes())
        assert r["seed"] == seed and r["policy_sha256"] == digest(args.ppo3)
        assert r["behavior_sha256"] == digest(folder/"behavior.npz")
        portable_receipts.append(dict(title=10, host="momentum", seed=seed, role="held",
            policy_sha256=r["policy_sha256"], file=str(folder/"behavior.npz"), sha256=r["behavior_sha256"],
            pixels=str(folder/"original-pixels.npz"), pixel_sha256=digest(folder/"original-pixels.npz"),
            steps=r["steps"], outcome="game_over" if r["game_over"] else "ending"))
    save(base / "held-ppo3/held-receipts.json", portable_receipts)
    pilot = {name:read_arm([base / ("held-"+name)], held) for name in ("ppo3","camera_sft")}
    save(base / "camera-sft-pilot.json", summarize(pilot, design, "ppo3"))
    draw(pilot, base / "camera-sft-pilot.png")
    ppo_model = checkpoint(ppo)
    if digest(ppo) == digest(sft):
        raise ValueError("The camera PPO checkpoint equals the SFT reference.")
    collect(args, "camera_ppo", ppo, held, base / "held-camera_ppo", 3, True)
    verify(base / "held-camera_ppo", ppo, ppo_model, held, "held")
    arms = {name:read_arm([base / ("held-"+name)], held) for name in ("ppo3","camera_sft","camera_ppo")}
    save(base / "paired.json", summarize(arms, design, "camera_sft"))
    draw(arms, base / "paired.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root","workers","ppo3"):
        parser.add_argument("--"+name, type=Path, required=True)
    run(parser.parse_args())
