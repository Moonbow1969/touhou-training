"""Run the preregistered TH10, TH11 and TH13 original full-credit cross-fit advantage comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import torch

from scripts.collect_th10_full_action import save
from scripts.package_th10_credit_logs import package
from scripts.package_th11_original_records import seal
from scripts.qualified_th11_stage2 import read_credit
from scripts.run_th10_camera_cohort import digest
from scripts.run_th10_camera_pair import run as paired_th10
from scripts.run_th10_high_rank_online import native_ledger
from scripts.run_th11_trial_credit import run as th11_game
from scripts.run_th13_accelerator_camera import CameraPool
from scripts.run_th13_four_game_generation2 import archive, wait
from scripts.score_th11_full_credit_pairs import score as plot_th11
from scripts.plot_th13_original_pair import run as plot_th13
from scripts.stage_native_ranked_evaluation import run as stage_results


def read(path):
    return json.loads(path.read_bytes())


def run(data, workers, title, six_game_retention=False):
    old = data / "five-game-generation3"
    study = data / ("six-game-retention" if six_game_retention else "five-game-crossfit-advantages")
    protocol = study / "camera-protocol.json"
    wait(protocol)
    plan = read(protocol)
    ref, alt = plan["reference_arm"], plan["candidate"]
    ref_sha = "b7d33148162c8a547a7cd9f49bca30d8e5f83053fb8a8c5f899d116ab5f57fd2"
    if six_game_retention:
        assert title == 10 and [ref, alt] == ["five-ppo3","six-ppo"]
        fit = data / "six-game-online/joint-six"
        assert plan["parent_actor"] == ref_sha
        assert digest(data / "six-game-online/protocol.json") == plan["candidate_registration"]
        assert digest(fit / "fit-protocol.json") == plan["candidate_fit"]
        number = 1690000
    else:
        assert [ref, alt] == ["five-ppo3","crossfit-ppo"] and plan["reference_sha256"] == ref_sha
        assert digest(old / "corrected/credits.json") == plan["training_manifest_sha256"]
        fit = study
        number = 1680000
    models = {ref:old / "policy/policy.ts", alt:fit / "policy/policy.ts"}
    assert digest(models[ref]) == ref_sha
    start = number + 100 * title
    seeds = plan["eval_seeds"][str(title)]
    assert seeds == list(range(start + 51, start + 67))
    ranks = {s:plan["rank_by_seed"][str(title)][str(s)] for s in seeds}
    assert all(r == (2 if s % 2 else 3) for s, r in ranks.items())
    physical = {str(s):"momentum" for s in seeds}
    orders = {}
    groups = {r:[s for s in seeds if ranks[s] == r] for r in (2, 3)}
    for subset in groups.values():
        for i, s in enumerate(subset):
            orders[str(s)] = [ref,alt] if i % 2 == 0 else [alt,ref]
    out = study / f"native{title}"
    out.mkdir(parents=True, exist_ok=True)
    path = study / ("th11-host-allocation.json" if title == 11 else f"th{title}-held-host-allocation.json")
    allocation = dict(study_protocol_sha256=digest(protocol),seed_host=physical,rank_lane=ranks,
        first_to_last_by_seed=orders,physical_pairing="One serial Momentum lane per rank. Same host, rank, game build, and controller seed for both full-stock credits. Alternating first actor within rank.")
    allocation["rank_lane"] = {str(s):r for s,r in ranks.items()}
    if path.exists():
        assert read(path) == allocation
    else:
        assert not any(out.rglob("receipt.json")) and not any(out.rglob("report.json"))
        save(path,allocation)
    previous = {10:old / "native10/ranked-native-comparison.json", 11:old / "th11-held-native-records.zip",
                13:data / "th13-visual-boss-curriculum/th13-held-native-logs.zip"}
    if six_game_retention:
        previous[10] = data / "five-game-crossfit-advantages/native10/ranked-native-comparison.json"
    wait(previous[title])
    if title == 11:
        qualification = study / "stage2-qualification.json"
        original = data / "th11-original-ppo/stage2-qualification.json"
        if not qualification.exists():
            shutil.copy2(original, qualification)
        assert digest(qualification) == digest(original)
    public = Path("results/2026-11-26-original-camera-training")
    def checked(model):
        wait(model.parent / "report.json")
        report = read(model.parent / "report.json")
        sha = digest(model)
        assert report["policy_sha256"] == sha
        if six_game_retention:
            if model == models[alt]:
                assert report["protocol_sha256"] == plan["candidate_fit"]
                assert report["manifest_sha256"] == digest(fit / "optimizer-input/credits.json")
        else:
            assert report["manifest_sha256"] == plan["training_manifest_sha256"]
            if model == models[alt]:
                assert report["actor_advantage_baselines_sha256"] == read(study / "fit-protocol.json")["actor_advantage_baselines_sha256"]
        return sha

    torch.set_num_threads(2)
    def evaluate(rank):
        subset = groups[rank]
        if title == 10:
            directory = out / ("hard" if rank == 2 else "lunatic")
            options = SimpleNamespace(out=directory,workers=workers,host="momentum",difficulty=rank,stage_sensor="qualified-v2-clock",
                reference_name="held-"+ref,candidate_name="held-"+alt,reference=models[ref],candidate=models[alt],seeds=subset,
                concurrency=1,candidate_history=False,validate_candidate=checked,interleave_arms=True)
            paired_th10(options)
            folders = [directory / ("held-"+arm) for arm in (ref,alt)]
            verified = native_ledger(folders,subset,rank)
            save(directory / "held-qualified-native-ranks.json",verified)
            package(study / f"th10-{directory.name}-held-native-logs.zip",folders)
            return verified
        for seed in subset:
            for arm in orders[str(seed)]:
                model = models[arm]
                sha = checked(model)
                folder = out / f"held-{arm}-{seed}"
                receipt = folder / "receipt.json"
                if not receipt.exists():
                    if title == 11:
                        th11_game(data,workers,seed,rank,role="held",arm=arm,model=model,out=folder,expected_sha=sha)
                    else:
                        pool = CameraPool(SimpleNamespace(host="momentum",host_label="momentum",known_hosts=None,
                            remote_root=f"/tmp/th13-v27-crossfit-168/rank{rank}",prior_bridge="/tmp/toho-th13-cadence-261125/bridge",
                            assets="/tmp/toho-th13-261125/assets/app",launcher="/tmp/toho-v26-high-camera/src/launch.sh",
                            image="toho-trial:endeavor-native-v6",python_executable="python3",checkpoint=model,output=out,
                            role="held",arm=arm,difficulty=rank))
                        pool.prepare()
                        pool.run_credit(seed)
                record = read(receipt)
                assert (record["role"],record["arm"],record["difficulty"],record["policy_sha256"],
                        record["host"],record["terminal"],record["continues"]) == ("held",arm,rank,sha,"momentum",True,0)
                if title == 11:
                    read_credit(folder)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(evaluate, (2,3)))
    if title == 10:
        save(out / "held-qualified-native-ranks.json", results[0] + results[1])
        stage_results(study, public, 10)
    elif title == 11:
        seal(study, "held", [(a,s) for s in seeds for a in (ref,alt)])
        plot_th11(study, public)
    else:
        archive(study, "held", seeds, [ref,alt], physical, {a:digest(models[a]) for a in (ref,alt)})
        for rank in ("hard","lunatic"):
            plot_th13(study,rank)
        stage_results(study, public, 13)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data",type=Path)
    parser.add_argument("workers",type=Path)
    parser.add_argument("title",type=int,choices=(10,11,13))
    parser.add_argument("--six-game-retention",action="store_true",help="TH10 contrast with the six-title joint update.")
    args = parser.parse_args()
    run(args.data.resolve(), args.workers.resolve(), args.title, args.six_game_retention)
