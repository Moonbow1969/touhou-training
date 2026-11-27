"""Run preregistered original TH10/13 matched tests of jointly trained vs averaged SFT."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.collect_th10_full_action import save
from scripts.package_th10_credit_logs import package
from scripts.run_th10_camera_cohort import digest
from scripts.run_th10_camera_pair import run as paired
from scripts.run_th10_high_rank_online import native_ledger
from scripts.run_th13_accelerator_camera import run as accelerator
from scripts.run_th13_four_game_generation2 import archive,wait
from scripts.summarize_th10_ranked_credits import run as summarize


def run(root,workers,title):
    root=root.resolve();workers=workers.resolve();s=root/"merged-four-native";native=s/f"native{title}"
    plan=json.loads((s/"camera-protocol.json").read_bytes())
    allocation=json.loads((s/f"th{title}-held-host-allocation.json").read_bytes())
    assert digest(s/"camera-protocol.json")==allocation["study_protocol_sha256"]
    ref,alt=plan["reference_arm"],plan["candidate"]
    paths={ref:root/"ranked-multigame-sft/four-title/policy.ts",alt:root/"specialist-vs-shared/merged-equal/policy.ts"}
    sha={a:digest(p) for a,p in paths.items()}
    assert sha[ref]==plan["behavior_sha256"] and sha[alt]==plan["candidate_sha256"]
    seeds=plan["eval_seeds"][str(title)];ranks=plan["rank_by_seed"][str(title)]
    assert len(set(seeds))==12
    if title==10:
        def group(rank):
            label="hard" if rank==2 else "lunatic";out=native/label
            selected=[k for k in seeds if ranks[str(k)]==rank];names=["held-"+ref,"held-"+alt]
            assert len(selected)==6 and all(allocation["seed_host"][str(k)]=="momentum" for k in selected)
            paired(SimpleNamespace(out=out,workers=workers,host="momentum",difficulty=rank,stage_sensor="qualified-v2-clock",
                reference_name=names[0],candidate_name=names[1],reference=paths[ref],candidate=paths[alt],
                seeds=selected,concurrency=1,candidate_history=False))
            dirs=[out/a for a in names];records=native_ledger(dirs,selected,rank)
            save(out/"qualified-stage-receipts.json",records);package(s/f"th10-{label}-held-native-logs.zip",dirs)
        with ThreadPoolExecutor(2) as pool:list(pool.map(group,(2,3)))
        summarize(s)
    else:
        wait(root/"four-game-online/th13-accelerator-held.json")
        for arm,index in ((ref,0),(alt,1),(alt,0),(ref,1)):
            for rank in (2,3):
                selected=[k for k in seeds if ranks[str(k)]==rank]
                assert len(selected)==6 and all(allocation["seed_host"][str(k)]=="accelerator" for k in selected)
                chosen=selected[index*3:(index+1)*3];missing=[k for k in chosen if not (native/f"held-{arm}-{k}/receipt.json").is_file()]
                if missing:
                    accelerator(SimpleNamespace(qualification=root/"engineering/th13-accelerator",connection_config=workers,
                        checkpoint=paths[arm],output=native,role="held",arm=arm,seeds=missing,workers=2,difficulty=rank))
        archive(s,"held",seeds,[ref,alt],allocation["seed_host"],sha)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True)
    p.add_argument("--workers",type=Path,required=True);p.add_argument("--title",type=int,choices=(10,13),required=True)
    a=p.parse_args();run(a.root,a.workers,a.title)
