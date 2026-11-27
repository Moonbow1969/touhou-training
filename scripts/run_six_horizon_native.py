"""Collect the preregistered TH10, TH11, and TH13 six-game PPO return-horizon originals."""
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
from scripts.plot_th13_original_pair import run as plot_th13
from scripts.qualified_th11_stage2 import read_credit
from scripts.run_th10_camera_cohort import checkpoint, collect, digest, verify
from scripts.run_th10_camera_pair import run as matched
from scripts.run_th10_high_rank_online import native_ledger
from scripts.run_th11_five_generation2 import validate_actor
from scripts.run_th11_trial_credit import run as th11_game
from scripts.run_th13_accelerator_camera import CameraPool, run as accelerator
from scripts.run_th13_four_game_generation2 import archive, wait
from scripts.score_th11_full_credit_pairs import score as plot_th11
from scripts.stage_native_ranked_evaluation import run as stage


def read(path):
    return json.loads(path.read_bytes())


def run(data, workers, title, endeavour, known):
    study = data / "six-game-return-horizon"
    protocol = study / "camera-protocol.json"
    wait(protocol)
    assert digest(protocol) == "781c2747519510bf1eb212ec2090679011c936907c861c702288a92409f5dd5c"
    plan = read(protocol)
    train_arm, ref, alt = (plan[k] for k in ("training_arm","reference_arm","candidate"))
    assert [train_arm, ref, alt] == ["six-ppo","six-gae985","six-return"]
    models = {a:study / a / "policy/policy.ts" for a in (ref,alt)}
    models[train_arm] = data / "six-game-online/joint-six/policy/policy.ts"
    behavior = digest(models[train_arm])
    assert behavior == plan["behavior_sha256"] == read(models[train_arm].parent / "report.json")["policy_sha256"]
    train = plan["train_seeds"][str(title)]
    held = plan["eval_seeds"][str(title)]
    start = 1710000 + title * 100
    assert train == list(range(start+1,start+33)) and held == list(range(start+51,start+67))
    ranks = {s:plan["rank_by_seed"][str(title)][str(s)] for s in train+held}
    assert all(r == (2 if s % 2 else 3) for s,r in ranks.items())
    native = study / f"native{title}"
    native.mkdir(exist_ok=True,parents=True)
    public = Path("results/2026-11-26-original-camera-training")
    hosts = {s:"momentum" for s in train+held}
    if title == 13:
        for group, each in ((train,8),(held,4)):
            for index, seed in enumerate(group):
                hosts[seed] = ("endeavour" if index < each else "momentum" if index < 2*each else "accelerator")
    order = {}
    for h in sorted(set(hosts.values())):
        for r in (2,3):
            subset = [s for s in held if hosts[s] == h and ranks[s] == r]
            assert len(subset) % 2 == 0
            order.update({str(s):([ref,alt] if i % 2 == 0 else [alt,ref]) for i,s in enumerate(subset)})
    allocation = dict(study_protocol_sha256=digest(protocol),training_policy_sha256=behavior,
        train_seed_host={str(s):hosts[s] for s in train},seed_host={str(s):hosts[s] for s in held},
        rank_lane={str(s):ranks[s] for s in train+held},first_to_last_by_seed=order,
        physical_pairing="Identical physical host and verified original build for each pair. Alternate arm order within physical host and rank. No Continue or assist.")
    allocation_name = "th11-host-allocation.json" if title == 11 else f"th{title}-held-host-allocation.json"
    af = study / allocation_name
    if af.is_file():
        assert read(af) == allocation
    else:
        assert not any(native.rglob("receipt.json")) and not any(native.rglob("report.json"))
        save(af,allocation)
    def checked(model):
        wait(model.parent / "report.json")
        report = read(model.parent / "report.json")
        fingerprint = digest(model)
        assert fingerprint == report["policy_sha256"]
        if model != models[train_arm]:
            fit = model.parent.parent / "fit-protocol.json"
            metadata = read(fit)
            assert report["protocol_sha256"] == digest(fit) and metadata["registered"] == digest(protocol)
            assert report["native_behavior_sha256"] == behavior and len(report["credits"]) == 192
        return fingerprint

    torch.set_num_threads(2)
    if title == 10:
        wait(data / "six-game-retention/native10/ranked-native-comparison.json")
        def th10(rank,role):
            selected = [s for s in (train if role == "train" else held) if ranks[s] == rank]
            folder = native / ("hard" if rank == 2 else "lunatic")
            folder.mkdir(exist_ok=True)
            options = SimpleNamespace(workers=workers, host="momentum", difficulty=rank, stage_sensor="qualified-v2-clock")
            if role == "train":
                model = models[train_arm]
                collect(options,train_arm,model,selected,folder,1,True)
                ledger = native_ledger([folder],selected,rank)
                receipts,_ = verify(folder,model,checkpoint(model),selected,role,host="momentum",difficulty=rank)
            else:
                options.out,options.seeds,options.concurrency = folder,selected,1
                options.reference_name,options.candidate_name = "held-"+ref,"held-"+alt
                options.reference,options.candidate = models[ref],models[alt]
                options.interleave_arms,options.candidate_history,options.validate_candidate = True,False,checked
                checked(models[ref])
                matched(options)
                dirs = [folder / ("held-"+a) for a in (ref,alt)]
                ledger,receipts = native_ledger(dirs,selected,rank),[]
                package(study / f"th10-{folder.name}-held-native-logs.zip",dirs)
            save(folder / (role+"-qualified-native-ranks.json"),ledger)
            return ledger,receipts
        with ThreadPoolExecutor(2) as pool:
            for role in ("train","held"):
                results = [f.result() for f in [pool.submit(th10,r,role) for r in (2,3)]]
                save(native / (role+"-qualified-native-ranks.json"),[r for rows,_ in results for r in rows])
                if role == "train":
                    save(native / "ppo-receipts.json",[r for _,rows in results for r in rows])
                    package(study / "th10-train-native-logs.zip",[native/"hard",native/"lunatic"])
        stage(study,public,10)
        return

    if title == 11:
        wait(data / "five-game-crossfit-advantages/th11-held-native-records.zip")
        source = data / "th11-original-ppo/stage2-qualification.json"
        qualification = study / "stage2-qualification.json"
        if not qualification.is_file():
            shutil.copy2(source,qualification)
        assert digest(qualification) == digest(source)
        actor = torch.jit.load(str(models[train_arm])).eval()
        def th11(seed, role, arm):
            model,rank = models[arm],ranks[seed]
            fingerprint = checked(model)
            folder = native / f"{role}-{arm}-{seed}"
            if not (folder / "receipt.json").exists():
                th11_game(data,workers,seed,rank,role=role,arm=arm,model=model,out=folder,expected_sha=fingerprint)
            r = read(folder / "receipt.json")
            assert (r["seed"],r["role"],r["arm"],r["host"],r["difficulty"],r["policy_sha256"]) == (seed,role,arm,hosts[seed],rank,fingerprint)
            read_credit(folder)
            if role == "train":
                return validate_actor(folder,actor,r)
        def th11_rank(rank,role):
            rows = []
            for s in train if role == "train" else held:
                if ranks[s] == rank:
                    for arm in ([train_arm] if role == "train" else order[str(s)]):
                        rows.append(th11(s,role,arm))
            return rows
        with ThreadPoolExecutor(2) as pool:
            for role in ("train","held"):
                results = [f.result() for f in [pool.submit(th11_rank,r,role) for r in (2,3)]]
                if role == "train":
                    save(study/"th11-train-value-reforward.json",dict(policy_sha256=behavior,rows=sorted(results[0]+results[1],key=lambda r:r["seed"])))
                    seal(study,role,[(train_arm,s) for s in train])
                else:
                    seal(study,role,[(a,s) for s in held for a in (ref,alt)])
        plot_th11(study,public)
        return

    assert title == 13 and endeavour and known
    barriers = {"momentum":data / "five-game-crossfit-advantages/th13-held-native-logs.zip",
        "endeavour":data / "five-game-generation3/th13-held-native-logs.zip",
        "accelerator":data / "engineering/th13-accelerator-concurrency/throughput.json"}
    def th13_credit(host,rank,role,arm,selected,concurrency):
        missing = [s for s in selected if not (native/f"{role}-{arm}-{s}/receipt.json").exists()]
        fingerprint = checked(models[arm])
        if missing:
            if host == "accelerator":
                accelerator(SimpleNamespace(qualification=data/"engineering/th13-accelerator",connection_config=workers,
                    checkpoint=models[arm],output=native,role=role,arm=arm,seeds=missing,workers=concurrency,difficulty=rank))
            else:
                remote = host == "endeavour"
                pool = CameraPool(SimpleNamespace(host=endeavour if remote else "momentum",host_label=host,
                    known_hosts=known if remote else None,remote_root=f"/tmp/th13-v27-six-horizon-171/rank{rank}",
                    prior_bridge=f"/tmp/toho-v26-quartet-camera/rank{rank}/bridge" if remote else "/tmp/toho-th13-cadence-261125/bridge",
                    assets="/tmp/toho-th13-261125/assets/app",launcher=("/tmp/toho-v26-quartet-camera" if remote else "/tmp/toho-v26-high-camera")+"/src/launch.sh",
                    image="toho-trial:endeavor-native-v6",python_executable="/tmp/toho-v26-torch/bin/python" if remote else "python3",
                    checkpoint=models[arm],output=native,role=role,arm=arm,difficulty=rank))
                pool.prepare()
                with ThreadPoolExecutor(concurrency) as lanes:
                    list(lanes.map(pool.run_credit,missing))
        for seed in selected:
            r = read(native/f"{role}-{arm}-{seed}/receipt.json")
            assert (r["role"],r["seed"],r["arm"],r["host"],r["difficulty"],r["policy_sha256"]) == (role,seed,arm,host,rank,fingerprint)
    def th13_host(host,role):
        wait(barriers[host])
        total = 2 if host != "accelerator" else min(4,read(barriers[host])["best_concurrency"])
        def per_rank(rank):
            subset = [s for s in (train if role == "train" else held) if ranks[s] == rank and hosts[s] == host]
            if role == "train":
                th13_credit(host,rank,role,train_arm,subset,max(1,total//2))
            else:
                for seed in subset:
                    for arm in order[str(seed)]:
                        th13_credit(host,rank,role,arm,[seed],1)
        with ThreadPoolExecutor(min(total,2)) as pool:
            list(pool.map(per_rank,(2,3)))
    with ThreadPoolExecutor(3) as pool:
        for role in ("train","held"):
            list(pool.map(lambda host:th13_host(host,role),["endeavour","momentum","accelerator"]))
            selected,arms = (train,[train_arm]) if role == "train" else (held,[ref,alt])
            archive(study,role,selected,arms,{str(s):hosts[s] for s in selected},{a:checked(models[a]) for a in arms})
    for label in ("hard","lunatic"):
        plot_th13(study,label)
    stage(study,public,13)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data",type=Path)
    parser.add_argument("workers",type=Path)
    parser.add_argument("title",type=int,choices=(10,11,13))
    parser.add_argument("--endeavour-host")
    parser.add_argument("--endeavour-known-hosts",type=Path)
    args = parser.parse_args()
    run(args.data.resolve(),args.workers.resolve(),args.title,args.endeavour_host,
        args.endeavour_known_hosts.resolve() if args.endeavour_known_hosts else None)
