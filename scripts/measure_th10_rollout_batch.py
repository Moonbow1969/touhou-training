"""Measure a sealed original TH10 batch from per-credit worker archive sizes and native clocks."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import shlex
import subprocess

import numpy as np
from scripts.collect_th10_full_action import save


REMOTE = """
import hashlib,json,sys
from pathlib import Path
root=Path.home()/'.local/share/toho-ml/th10-official-trial-002a/full-story'
rows=[]
for job in json.load(sys.stdin):
    candidates=[]
    for path in sorted(root.glob(job['prefix']+'-[123]')):
        archive=path/'behavior.tar.gz';status=path/'status'
        if not archive.is_file() or not status.is_file() or status.read_text().strip()!='0':continue
        sha=hashlib.sha256(archive.read_bytes()).hexdigest()
        if sha in job['rejected']:continue
        candidates.append(dict(seed=job['seed'],compressed_payload_sha256=sha,payload_bytes=archive.stat().st_size,
            claimed_utc=(path/'claimed').stat().st_mtime,archive_ready_utc=archive.stat().st_mtime))
    if len(candidates)!=1:raise ValueError((job['seed'],len(candidates)))
    rows.extend(candidates)
print(json.dumps(rows))
"""


def run(directory, workers, out):
    jobs=[];receipts={};native={}
    for p in sorted(directory.glob("*/report.json")):
        receipt=json.loads(p.read_bytes());seed=receipt["seed"]
        if seed in receipts or not receipt["terminal"]:raise ValueError("Duplicate or censored original credit")
        receipts[seed]=receipt
        prefix=f"{seed}-{receipt['policy_sha256'][:12]}"
        if receipt["requested_hold_ms"]!=120:prefix+=f"-h{receipt['requested_hold_ms']}"
        if receipt["source"].get("difficulty",1)==0:prefix+="-easy"
        if receipt["stage_sensor"]=="qualified-v2":prefix+="-stagev2"
        rejected=[json.loads(q.read_bytes())['archive_sha256'] for q in p.parent.glob('rejected-*.json')]
        jobs.append(dict(seed=seed,prefix=prefix,rejected=rejected))
        with gzip.open(p.parent/'native.csv.gz','rt') as f:times=[int(r['utc_ms'])/1000 for r in csv.DictReader(f)]
        native[seed]=dict(first_native_utc=times[0],last_native_utc=times[-1],
            decision_count=receipt['steps'],local_credit_bytes=sum(q.stat().st_size for q in p.parent.iterdir() if q.is_file()))
    hosts={r['host'] for r in receipts.values()}
    if len(hosts)!=1 or not jobs:raise ValueError('Measure one sealed batch on one worker host')
    host=hosts.pop();command=json.loads(workers.read_bytes())[host]['command'][:-1]
    payload=json.dumps(jobs).encode()
    stats=json.loads(subprocess.check_output([*command,'python3 -c '+shlex.quote(REMOTE)],input=payload,timeout=120))
    for row in stats:
        receipt=receipts[row['seed']]
        if receipt.get('compressed_ssh_payload_sha256',row['compressed_payload_sha256'])!=row['compressed_payload_sha256']:
            raise ValueError('Transferred archive differs from worker copy')
        row.update(native[row['seed']])
    wall=max(r['archive_ready_utc'] for r in stats)-min(r['claimed_utc'] for r in stats)
    active=sum(r['last_native_utc']-r['first_native_utc'] for r in stats)
    steps=sum(r['decision_count'] for r in stats);wire=sum(r['payload_bytes'] for r in stats)
    save(out,dict(physical_host=host,credits=len(stats),decisions=steps,worker_wall_s=wall,active_credit_wall_s=active,
        effective_concurrent_gameplay=active/wall,decisions_per_worker_wall_s=steps/wall,
        median_claim_to_first_native_s=float(np.median([r['first_native_utc']-r['claimed_utc'] for r in stats])),
        transfer_payload_bytes=wire,required_average_payload_mbit_s=8*wire/wall/1e6,
        local_bytes=sum(r['local_credit_bytes'] for r in stats),rows=stats,
        interpretation='One gzip-compressed SSH download per whole credit, excluding headers, retransmission and model deployment. Worker wall covers first claim to final compressed archive. Effective concurrency includes setup/teardown gaps. Worker clocks and original native UTC are from the same physical host.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('directory','workers','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();run(args.directory,args.workers,args.out)
