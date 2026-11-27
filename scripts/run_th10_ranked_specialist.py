"""Evaluate two frozen human-replay SFT policies on original TH10 Hard and Lunatic."""
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


def run(root,workers):
    root=root.resolve();study=root/'ranked-multigame-sft/held-th10'
    plan=json.loads((study/'protocol.json').read_bytes())
    ref,alt=plan['reference_arm'],plan['candidate'];models={n:root/p for n,p in plan['arms'].items()}
    assert all(digest(path)==plan['checkpoints'][arm] for arm,path in models.items())
    def group(rank):
        label={2:'hard',3:'lunatic'}[rank]
        out=study/'native10'/label;out.mkdir(exist_ok=True,parents=True)
        seeds=[s for s in plan['eval_seeds']['10'] if plan['rank_by_seed']['10'][str(s)]==rank]
        assert len(set(seeds))==12 and all(3-s%2==rank for s in seeds)
        arms=('held-'+ref,'held-'+alt)
        paired(SimpleNamespace(out=out,workers=workers.resolve(),host='momentum',difficulty=rank,stage_sensor='qualified-v2',
            reference_name=arms[0],candidate_name=arms[1],reference=models[ref],candidate=models[alt],
            seeds=seeds,concurrency=1,candidate_history=False))
        folders=[out/a for a in arms];records=native_ledger(folders,seeds,rank)
        save(out/'qualified-stage-receipts.json',records)
        package(study/f'th10-{label}-native-logs.zip',folders)
        return records
    with ThreadPoolExecutor(2) as pool:groups=list(pool.map(group,(2,3)))
    save(study/'native10/qualified-stage-receipts.json',[r for group in groups for r in group])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=Path,required=True)
    args=p.parse_args();run(args.root,args.workers)
