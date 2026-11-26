"""Build a separate training-only corpus from segmented official-trial contact runs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from shutil import disk_usage
from threading import Lock
from types import SimpleNamespace

from scripts import collect_trial_contact_windows as trial
from scripts.checkpoint_storage import digest
from scripts.train_full_policy import publish_json


def specification(args, deployed, reserve):
    reference = reserve["workers"]
    if reserve["status"] != "complete" or sum(w["counts"].get("accepted_windows", 0) for w in reference.values()) < 4096:
        raise ValueError("The independent reserved evaluation must be complete before training acquisition.")
    if set(args.workers) != set(reference):
        raise ValueError("The training acquisition uses the same qualified lanes.")
    for name in args.workers:
        frozen = reference[name]["launch"]
        if deployed[name]["status"] != "ready" or any(deployed[name][key] != value for key, value in frozen.items()):
            raise ValueError(f"Native deployment changed for {name}.")
    return dict(schema=1, purpose="training_only", cohort="randomized held-12-tick native injury in official TH14 trial 0.01b",
                reserved_eval_collection_sha256=digest(args.reserve / "collection.json"),
                reserved_eval_window_shas=sorted(w["shard_sha256"] for w in reference.values()),
                reserved_eval_seeds=sorted(w["seed"] for w in reference.values()),
                collector_sha256=digest(Path(trial.__file__)), supervisor_sha256=digest(Path(__file__)),
                target_windows=len(args.workers)*args.chunks*args.windows, minimum_windows=args.minimum_windows,
                lane_order=args.workers, chunks=args.chunks, windows_per_chunk=args.windows, steps_per_chunk=args.steps,
                episodes_per_chunk=args.episodes, sampling=reserve["sampling"], seed_base=args.seed,
                maximum_uncompressed_bytes_per_shard=args.shard_mib * (1024**2), local_disk_reserve_gib=8)


def collect(args, deployed, reserved):
    spec = specification(args, deployed, reserved)
    host = dict(item.split("=", 1) for item in args.worker_host)
    if set(host) != set(args.workers) or len(set(host.values())) < 2:
        raise ValueError("Record the physical host assignment for every lane.")
    path = args.out / "manifest.json"
    if args.resume:
        manifest = json.loads(path.read_text())
        if (manifest["specification"], manifest["worker_host"]) != (spec, host):
            raise ValueError("The resume request changes the separately frozen training acquisition.")
        for row in manifest["shards"].values():
            if digest(args.out / row["relative_path"]) != row["shard_sha256"]:
                raise ValueError("A completed training shard has been changed.")
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        manifest = dict(specification=spec, worker_host=host, shards={}, attempts={})
    lock = Lock()

    def update(status):
        shards = manifest["shards"].values()
        totals = dict(windows=sum(row["counts"].get("accepted_windows", 0) for row in shards),
                      injuries=sum(row["counts"].get("injuries", 0) for row in shards),
                      decisions=sum(row["sampled_steps"] for row in shards),
                      bytes=sum(row["shard_bytes"] for row in shards),
                      committed_shards=len(shards))
        manifest.update(status=status, totals=totals)
        publish_json(path, manifest)
        return totals

    def lane(worker, rank):
        for index in range(1, args.chunks + 1):
            key = f"chunk-{index:02}/{worker}"
            if key in manifest["shards"]:
                continue
            if disk_usage(args.out).free < 8 * trial.GIB:
                raise OSError("Disk reserve reached before a training shard launch.")
            directory = args.out / f"chunk-{index:02}"
            directory.mkdir(exist_ok=True)
            final = directory / f"{worker}.jsonl.gz"
            stage_args = SimpleNamespace(out=directory, windows=args.windows, steps=args.steps, episodes=args.episodes,
                                         frame_limit=spec["sampling"]["frame_limit"],
                                         near_radius=spec["sampling"]["near_projectile_radius"],
                                         max_uncompressed_bytes=spec["maximum_uncompressed_bytes_per_shard"])
            for _ in range(args.attempts):
                with lock:
                    attempts = manifest["attempts"].setdefault(key, [])
                    attempt = len(attempts)
                    seed = args.seed + rank * 1009 + index * 10007 + attempt * 1000003
                    if seed in spec["reserved_eval_seeds"]:
                        raise ValueError("A training seed overlaps the reserved evaluation.")
                    if final.exists():
                        final.rename(directory / f"{worker}.abandoned-{attempt:03}.jsonl.gz")
                    entry = dict(seed=seed, status="started")
                    attempts.append(entry)
                    update("collecting")
                try:
                    row = trial.capture(worker, deployed[worker], stage_args, seed)
                    if row["counts"].get("accepted_windows") != args.windows:
                        raise RuntimeError(f"Chunk did not achieve its window quota: {row['counts']}.")
                except Exception as error:
                    with lock:
                        entry.update(status="error", error=f"{type(error).__name__}: {error}")
                        update("collecting")
                    print(json.dumps(dict(chunk=index, worker=worker, event="chunk_retry", seed=seed)), flush=True)
                    continue
                with lock:
                    entry["status"] = "complete"
                    manifest["shards"][key] = dict(row, physical_host=host[worker], chunk=index,
                                                   relative_path=final.relative_to(args.out).as_posix())
                    totals = update("collecting")
                print(json.dumps(dict(chunk=index, worker=worker, event="shard_committed", **totals)), flush=True)
                break

    update("collecting")
    with ThreadPoolExecutor(max_workers=len(args.workers)) as pool:
        jobs = [pool.submit(lane, worker, rank) for rank, worker in enumerate(args.workers)]
        for job in jobs:
            job.result()
    status = "complete" if manifest["totals"]["windows"] >= args.minimum_windows else "partial"
    print(json.dumps(dict(status=status, **update(status))), flush=True)
    if status == "partial":
        raise RuntimeError("The bounded corpus fell below its minimum window target. Completed shards are preserved.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers-config", type=Path, required=True)
    parser.add_argument("--reserve", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", nargs="+", required=True)
    parser.add_argument("--worker-host", action="append", required=True, metavar="WORKER=HOST")
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--windows", type=int, default=850)
    parser.add_argument("--minimum-windows", type=int, default=20000)
    parser.add_argument("--steps", type=int, default=14000)
    parser.add_argument("--episodes", type=int, default=96)
    parser.add_argument("--shard-mib", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1970031)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (min(args.chunks, args.windows, args.minimum_windows, args.steps, args.episodes, args.shard_mib, args.attempts) < 1
            or args.minimum_windows > len(args.workers) * args.chunks * args.windows
            or len(args.workers) != len(set(args.workers)) or any(not lane.isalpha() for lane in args.workers)
            or any(item.count("=") != 1 for item in args.worker_host)):
        parser.error("Use positive bounded quotas, distinct safe lane labels, and explicit host groups.")
    deployed = json.loads(args.workers_config.read_text())
    reserved = json.loads((args.reserve / "collection.json").read_text())
    collect(args, deployed, reserved)


if __name__ == "__main__":
    main()
