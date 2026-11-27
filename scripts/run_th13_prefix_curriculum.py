"""Collect up to twelve independently registered live TH13 human-prefix attempts and ordinary openings."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.audit_th13_policy_suffix import verify
from scripts.collect_th10_full_action import save
from scripts.qualify_th13_human_prefix import run as prefix_credit
from scripts.run_th13_accelerator_camera import CameraPool
from scripts.run_th10_camera_cohort import digest


def run(root,mode):
    root=root.resolve();study=root/"human-prefix-curriculum"
    native=study/"native13"; control=study/"opening-control"
    native.mkdir(parents=True,exist_ok=True);control.mkdir(parents=True,exist_ok=True)
    policy=root/"four-game-online/policy/policy.ts"
    assert digest(policy)=="3b26d1b7f8a8a27c5e271bc1e1eccf8e9c58d9f4c3186cc168efaacb3f07b82c"
    protocol=dict(title=13,rank=2,character=0,host="momentum",build="official TH13 0.01a trial",
        source_policy_sha256=digest(policy),human_train_group="Taraka2",human_input_sha256="d8e38081c8fd646dff3bc95f99478a3b90a27ee27947d6c4d2883a73d27486b9",
        target_native_stage1_tick=3000,fresh_native_rng=True,opening_spares=2,opening_bombs=2,continue_allowed=False,
        attempt_seeds=list(range(1281301,1281313)),opening_control_seeds=list(range(1281321,1281333)),
        policy_dataset="native13/train-humanprefix-visual-<seed>/rollout.npz contains post-handoff observations/actions only",
        ordinary_opening="opening-control/train-four-ppo-<seed>/rollout.npz",qualification="eligible.json in each folder; original Hard Reimu trace, consumed controls, neural logp/value rerun, rank-screen Stage independently verified.",
        before_handoff_failures="Every attempted seed retained; earlier Game Over has no policy rows. No technical retries in prefix cohort.",
        primary_training_plan="Use at least 4 qualifying suffixes in seed order; same optimizer and actual transition budget for ordinary-opening control. Evaluate from fresh ordinary openings.")
    path=study/"protocol.json"
    if path.exists():assert json.loads(path.read_bytes())==protocol
    else:save(path,protocol)
    if mode=="register":return
    pool=None
    if mode=="opening":
        o=SimpleNamespace(host="momentum",host_label="momentum",known_hosts=None,remote_root="/tmp/toho-v26-prefix-comparison-opening",
             prior_bridge="/tmp/toho-th13-cadence-261125/bridge",assets="/tmp/toho-th13-261125/assets/app",launcher="/tmp/toho-v26-high-camera/src/launch.sh",
             image="toho-trial:endeavor-native-v6",python_executable="python3",checkpoint=policy,output=control,role="train",arm="four-ppo",difficulty=2)
        pool=CameraPool(o);pool.prepare()
    def credit(seed):
        folder=(native/f"train-humanprefix-visual-{seed}" if mode=="prefix" else control/f"train-four-ppo-{seed}")
        if (folder/"eligible.json").exists():return
        try:
            if mode=="prefix":
                if not (folder/"qualification.json").exists():prefix_credit(root,seed,3000,native)
                if not json.loads((folder/"qualification.json").read_bytes())["validated_handoff"]:return
            else:pool.run_credit(seed)
            result=verify(folder,policy,prefix=mode=="prefix");save(folder/"eligible.json",result)
            print("ELIGIBLE",mode,seed,result["steps"],flush=True)
        except Exception as e:
            folder.mkdir(parents=True,exist_ok=True)
            save(folder/"collection-error.json",dict(seed=seed,error=repr(e)))
            print("ERROR",mode,seed,repr(e),flush=True)
    seeds=protocol["attempt_seeds" if mode=="prefix" else "opening_control_seeds"]
    with ThreadPoolExecutor(1) as ex:list(ex.map(credit,seeds))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True);parser.add_argument("--mode",choices=["prefix","opening","register"],required=True)
    a=parser.parse_args();run(a.root,a.mode)
