"""Finish and verify the assigned two-host TH13 generation-2 PPO evaluation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from types import SimpleNamespace
import zipfile

import numpy as np
from scripts.collect_th10_full_action import save
from scripts.read_th13_trial_ranking import scan
from scripts.run_th10_camera_cohort import digest
from scripts.run_th13_accelerator_camera import CameraPool, run as accelerator
from scripts.score_original_camera_run import verify
from scripts.score_native_stage_clock import summarize as clock_score


def wait(path):
    while not path.is_file():time.sleep(4)


def archive(study, role, seeds, arms, hosts, hashes):
    native=study/"native13";out=study/f"th13-{role}-evaluation"
    for name in ("endpoints","action-traces"):(out/name).mkdir(parents=True,exist_ok=True)
    pictures={Path(r["path"]).parent.name:r for r in scan(native)["rows"]}
    archive_path=study/f"th13-{role}-native-logs.zip";temp=archive_path.with_suffix(".inprogress")
    rows=[]
    with zipfile.ZipFile(temp,"w",zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for seed in seeds:
            for arm in arms:
                folder=native/f"{role}-{arm}-{seed}";receipt=folder/"receipt.json";r=json.loads(receipt.read_bytes())
                expected_rank=3-seed%2;data=np.fromfile(folder/"steps.bin","<f4").reshape(-1,143)
                assert r["host"]==hosts[str(seed)] and r["policy_sha256"]==hashes[arm] and r["difficulty"]==expected_rank
                assert r["seed"]==seed and r["terminal"] and r["continues"]==0 and r["steps"]==len(data)
                assert list(data[0,[125,127,129]])==[1,2,2] and (data[:,141]==expected_rank).all()
                assert digest(folder/"steps.bin")==r["native_difficulty_trace_sha256"]
                v=verify(13,receipt,r,folder/"rollout.npz",arm,out)
                v.update(difficulty=expected_rank,score=int(data[-1,131])*10,native_input_matches=r["native_key_match"],
                         native_rank_rows=len(data),ranking_stage=pictures[folder.name]["readout"],
                         ranking_agrees=pictures[folder.name]["matches_native"],native_rank_trace_sha256=r["native_difficulty_trace_sha256"])
                rows.append(v)
                for f in ("steps.bin","receipt.json","ranking.png","selected-difficulty.png","events.tsv","final-stock.bin"):
                    z.write(folder/f,f"{folder.name}/{f}")
        save(out/"credits.json",rows);z.write(out/"credits.json","credits.json")
        if role=="held":
            scores={}
            for rank,label in ((2,"hard"),(3,"lunatic")):
                subset=[r for r in rows if r["difficulty"]==rank]
                contrast=clock_score(subset,13,arms)
                stats={}
                for a in arms:
                    s=[r for r in subset if r["arm"]==a]
                    stats[a]=dict(n=len(s),stage2=sum(r["stage2_reached"] for r in s),
                        stage2_resources=[dict(seed=r["seed"],spares=r["stage2_spares"],bombs=r["stage2_bombs"]) for r in s if r["stage2_reached"]],
                        mean_terminal_bombs=float(np.mean([r["terminal_bombs"] for r in s])),total_casts=sum(r["effective_bombs"] for r in s),
                        mean_native_score=float(np.mean([r["score"] for r in s])),terminal_spares=[r["terminal_spares"] for r in s],
                        ranking_matches=sum(r["ranking_agrees"] is True for r in s))
                scores[label]=dict(stage1_tick=contrast,groups=stats)
            save(out/"paired-by-rank.json",scores);z.write(out/"paired-by-rank.json","paired-by-rank.json")
        z.write(study/"th13-held-host-allocation.json","host-allocation.json")
    with zipfile.ZipFile(temp) as z:assert z.testzip() is None
    temp.replace(archive_path)


def run(root,workers):
    root=root.resolve();study=root/"four-game-ppo-generation2";native=study/"native13"
    plan=json.loads((study/"camera-protocol.json").read_bytes())
    assignment=json.loads((study/"th13-held-host-allocation.json").read_bytes())
    assert assignment["study_protocol_sha256"]==digest(study/"camera-protocol.json")
    ref,alt=plan["reference_arm"],plan["candidate"];train=plan["train_seeds"]["13"];held=plan["eval_seeds"]["13"]
    parent=root/"four-game-online/policy/policy.ts";candidate=study/"policy/policy.ts"
    sha=digest(parent);assert sha==plan["behavior_sha256"]
    def collect(host,rank,arm,path,seeds):
        missing=[s for s in seeds if not (native/f"held-{arm}-{s}/receipt.json").is_file()]
        if not missing:return
        if host=="accelerator":
            accelerator(SimpleNamespace(qualification=root/"engineering/th13-accelerator",connection_config=workers,
                checkpoint=path,output=native,role="held",arm=arm,seeds=missing,workers=3,difficulty=rank))
        else:
            options=SimpleNamespace(host="momentum",host_label="momentum",known_hosts=None,remote_root=f"/tmp/toho-v26-four-g2/rank{rank}",
                prior_bridge="/tmp/toho-th13-cadence-261125/bridge",assets="/tmp/toho-th13-261125/assets/app",
                launcher="/tmp/toho-v26-high-camera/src/launch.sh",image="toho-trial:endeavor-native-v6",python_executable="python3",
                checkpoint=path,output=native,role="held",arm=arm,difficulty=rank)
            actor=CameraPool(options);actor.prepare()
            with ThreadPoolExecutor(2) as pool:list(pool.map(actor.run_credit,missing))
    def host_run(host):
        if host=="accelerator":wait(root/"four-game-online/th13-accelerator-held.json")
        else:
            for s in train:wait(native/f"train-{ref}-{s}/receipt.json")
            archive(study,"train",train,[ref],assignment["train_seed_host"],{ref:sha})
        assigned=[s for s in held if assignment["seed_host"][str(s)]==host]
        assert len(assigned)==12
        for arm,n in ((ref,0),(alt,1),(alt,0),(ref,1)):
            path=parent if arm==ref else candidate
            if arm==alt:
                wait(candidate.parent/"report.json")
                assert digest(candidate)==json.loads((candidate.parent/"report.json").read_bytes())["policy_sha256"]
            for rank in (2,3):
                seeds=[s for s in assigned if plan["rank_by_seed"]["13"][str(s)]==rank]
                assert len(seeds)==6
                collect(host,rank,arm,path,seeds[n*3:(n+1)*3])
    with ThreadPoolExecutor(2) as pool:list(pool.map(host_run,("momentum","accelerator")))
    archive(study,"held",held,[ref,alt],assignment["seed_host"],{ref:sha,alt:digest(candidate)})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True);p.add_argument("--workers",type=Path,required=True)
    a=p.parse_args();run(a.root,a.workers.resolve())
