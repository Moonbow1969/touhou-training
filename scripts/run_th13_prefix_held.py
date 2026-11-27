"""Evaluate two independently fitted TH13 policies from matched ordinary Hard openings."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from types import SimpleNamespace

from scripts.collect_th10_full_action import save
from scripts.run_th10_camera_cohort import digest
from scripts.run_th13_accelerator_camera import run as collect
from scripts.run_th13_four_game_generation2 import archive, wait


def run(root,workers):
    root=root.resolve();study=root/"human-prefix-curriculum";native=study/"native13"
    src=study/"camera-protocol.json";protocol=json.loads(src.read_bytes())
    seeds=protocol["eval_seeds"]["13"];ref,alt=protocol["reference_arm"],protocol["candidate"]
    assert seeds==list(range(1281351,1281363)) and list(set(protocol["rank_by_seed"]["13"].values()))==[2]
    hosts={str(s):"accelerator" for s in seeds}
    allocation=study/"th13-held-host-allocation.json"
    expected=dict(study_protocol_sha256=digest(src),seed_host=hosts,order_by_seed={str(s):([ref,alt] if s%2 else [alt,ref]) for s in seeds},
        physical_pairing="All complete credits on Accelerator; arm order alternates by controller seed, no human prefix, ordinary original opening.")
    if allocation.exists():assert json.loads(allocation.read_bytes())==expected
    else:
        assert not any(native.glob("held-*/receipt.json"))
        save(allocation,expected)
    hashes={}
    for arm in (ref,alt):
        model=study/arm/"policy/policy.ts"; report=model.parent/"report.json";wait(report)
        h=digest(model);assert h==json.loads(report.read_bytes())["policy_sha256"]; hashes[arm]=h
    # Finish the prior weight-merge wave on this same host before borrowing its lane.
    wait(root/"merged-four-native/th13-held-evaluation/paired-by-rank.json")
    # Keep the registered within-pair order, using two released private games.
    # A single deployment per arm wave avoids concurrent writes to the same runtime.
    for arm,subset in ((ref,seeds[::2]),(alt,seeds[1::2]),(alt,seeds[::2]),(ref,seeds[1::2])):
        missing=[s for s in subset if not (native/f"held-{arm}-{s}/receipt.json").is_file()]
        if missing:
            collect(SimpleNamespace(qualification=root/"engineering/th13-accelerator",connection_config=workers,
                checkpoint=study/arm/"policy/policy.ts",output=native,role="held",arm=arm,seeds=missing,workers=2,difficulty=2))
    archive(study,"held",seeds,[ref,alt],hosts,hashes,protocol["rank_by_seed"]["13"])


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True);parser.add_argument("--workers",type=Path,required=True)
    a=parser.parse_args();run(a.root,a.workers.resolve())
