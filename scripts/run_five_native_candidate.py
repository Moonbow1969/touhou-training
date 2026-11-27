"""Run only five-game PPO candidates and verify they pair to pre-existing original SFT credits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace
from scripts.collect_th10_full_action import save
from scripts.run_th10_camera_cohort import digest,collect,checkpoint,verify
from scripts.run_th10_high_rank_online import native_ledger
from scripts.package_th10_credit_logs import package
from scripts.run_th13_accelerator_camera import CameraPool,run as accelerator
from scripts.run_th13_four_game_generation2 import wait
from scripts.run_th11_trial_credit import run as th11
from scripts.qualified_th11_stage2 import read_credit

def run(root,workers,title,ssh_host,known):
    study=root/"five-game-online";d=study/f"native{title}";d.mkdir(exist_ok=True,parents=True)
    old=root/("th11-original-ppo" if title==11 else "four-game-merged-online")
    protocol=study/"camera-protocol.json";p=json.loads(protocol.read_bytes());arm=p["candidate"];assert arm=="five-ppo"
    seeds=p["eval_seeds"][str(title)];assert len(seeds)==16 and len(set(seeds))==16
    ranks=p["rank_by_seed"][str(title)];assert all(ranks[str(s)]==3-s%2 for s in seeds)
    model=study/"policy/policy.ts"
    if title==13:
        old_hosts=json.loads((old/"th13-held-host-allocation.json").read_bytes())["seed_host"]
        hosts={str(s):old_hosts[str(s)] for s in seeds}
    else:hosts={str(s):"momentum" for s in seeds}
    alloc=study/f"th{title}-held-host-allocation.json";reg=dict(seed_host=hosts,study_protocol_sha256=digest(protocol),
        prior_source=old.name,reused_reference="Original same-seed merged human SFT credits already collected; this execution launches only the five-game candidate.",
        physical_pairing="Same physical host, original build, controller seed, difficulty, character, cadence and qualified runtime. The baselines are historical frozen controls; order relative to the candidate is not randomized.")
    if alloc.exists():assert json.loads(alloc.read_bytes())==reg
    else:save(alloc,reg)
    wait(model.parent/"report.json");sha=digest(model);assert sha==json.loads((model.parent/"report.json").read_bytes())["policy_sha256"]
    def original(s,rank):
        if title==10:return old/f"native10/{'hard' if rank==2 else 'lunatic'}/held-merged-sft/momentum-{s}/report.json"
        return old/f"native{title}/held-merged-sft-{s}/receipt.json"
    def check(s,rank):
        f=original(s,rank);wait(f);r=json.loads(f.read_bytes())
        assert r["seed"]==s and r["difficulty"]==rank and r["host"]==hosts[str(s)] and r["policy_sha256"]==p["behavior_sha256"]
    def th10_group(rank):
        chosen=[s for s in seeds if ranks[str(s)]==rank];folder=d/("hard" if rank==2 else "lunatic")/("held-"+arm)
        for s in chosen:check(s,rank)
        collect(SimpleNamespace(workers=workers,host="momentum",difficulty=rank,stage_sensor="qualified-v2-clock"),"held-"+arm,model,chosen,folder,1,True)
        verify(folder,model,checkpoint(model),chosen,"held",host="momentum",difficulty=rank)
        rows=native_ledger([folder],chosen,rank,"momentum")
        save(folder.parent/"five-ppo-qualified-native.json",rows);return rows
    def th11_group(rank):
        # Preserve the same existing sequential queue for each original rank.
        for s in seeds:
            if ranks[str(s)]==rank:
                for older_arm in ["merged-sft","th11-ppo"]:wait(old/f"native11/held-{older_arm}-{s}/receipt.json")
        for s in seeds:
            if ranks[str(s)]!=rank:continue
            check(s,rank);folder=d/f"held-{arm}-{s}"
            if not (folder/"receipt.json").exists():th11(root,workers,s,rank,role="held",arm=arm,model=model,out=folder,expected_sha=sha)
            r=read_credit(folder);assert r["seed"]==s
    def th13_host(physical):
        chosen=[s for s in seeds if hosts[str(s)]==physical]
        # Preserve prior physical queues; this allows Accelerator's completed half to start immediately.
        peers=json.loads((old/"th13-held-host-allocation.json").read_bytes())["seed_host"]
        for key,h in peers.items():
            if h==physical:
                for a in ["merged-sft","merged-ppo"]:wait(old/f"native13/held-{a}-{key}/receipt.json")
        for rank in [2,3]:
            subset=[s for s in chosen if ranks[str(s)]==rank]
            for s in subset:check(s,rank)
            missing=[s for s in subset if not (d/f"held-{arm}-{s}/receipt.json").exists()]
            if not missing:continue
            if physical=="accelerator":
                accelerator(SimpleNamespace(qualification=root/"engineering/th13-accelerator",connection_config=workers,checkpoint=model,output=d,role="held",arm=arm,seeds=missing,workers=2,difficulty=rank))
            else:
                pool=CameraPool(SimpleNamespace(host=ssh_host,host_label="endeavour",known_hosts=known,remote_root=f"/tmp/toho-v26-merged-online/rank{rank}",
                    prior_bridge=f"/tmp/toho-v26-quartet-camera/rank{rank}/bridge",assets="/tmp/toho-th13-261125/assets/app",
                    launcher="/tmp/toho-v26-quartet-camera/src/launch.sh",image="toho-trial:endeavor-native-v6",python_executable="/tmp/toho-v26-torch/bin/python",
                    checkpoint=model,output=d,role="held",arm=arm,difficulty=rank))
                pool.prepare()
                with ThreadPoolExecutor(2) as lanes:list(lanes.map(pool.run_credit,missing))
    with ThreadPoolExecutor(2) as executor:
        if title==10:
            rows=list(executor.map(th10_group,(2,3)));save(study/"th10-candidate-qualified.json",rows[0]+rows[1])
            package(study/"th10-candidate-native-logs.zip",[d/"hard"/("held-"+arm),d/"lunatic"/("held-"+arm)])
        elif title==11:list(executor.map(th11_group,(2,3)))
        else:list(executor.map(th13_host,sorted(set(hosts.values()))))
    pairs=[]
    for s in seeds:
        rank=ranks[str(s)];reference=original(s,rank)
        candidate=(d/("hard" if rank==2 else "lunatic")/("held-"+arm)/f"momentum-{s}/report.json") if title==10 else d/f"held-{arm}-{s}/receipt.json"
        j=json.loads(candidate.read_bytes());assert j["policy_sha256"]==sha and j["host"]==hosts[str(s)] and j["difficulty"]==rank
        pairs.append(dict(seed=s,rank=rank,host=j["host"],reference=str(reference.relative_to(root)).replace("\\","/"),candidate=str(candidate.relative_to(root)).replace("\\","/"),
            reference_receipt_sha256=digest(reference),candidate_receipt_sha256=digest(candidate)))
    save(study/f"th{title}-candidate-pairing.json",pairs)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("workers",type=Path);p.add_argument("title",type=int,choices=(10,11,13));p.add_argument("--ssh-host");p.add_argument("--known-hosts",type=Path)
    a=p.parse_args();run(a.root.resolve(),a.workers.resolve(),a.title,a.ssh_host,a.known_hosts.resolve() if a.known_hosts else None)
