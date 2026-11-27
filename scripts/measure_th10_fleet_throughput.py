"""Account for LAN rollout fleet throughput at original full-credit boundaries."""
import argparse
import gzip
import csv
import json
from pathlib import Path
import shlex
import subprocess
import time

import numpy as np


def audit(cohort,private):
    manifest=json.loads((cohort/'ppo-receipts.json').read_bytes())
    fleet=json.loads(private.read_bytes());all_credits=[];clocks={}
    code='''from pathlib import Path
import json,time
base=Path.home()/'.local/share/toho-ml/th10-official-trial-002a/full-story'
out={}
for seed,sha in JOBS:
    scans=[]
    for folder in sorted(base.glob(f'{seed}-{sha[:12]}-*')):
        if not (folder/'claimed').is_dir() or not (folder/'status').is_file():continue
        status=int((folder/'status').read_text().strip())
        archive=folder/'behavior.tar.gz'
        scans.append(dict(attempt=folder.name.rsplit('-',1)[-1],status=status,claimed_ns=(folder/'claimed').stat().st_mtime_ns,
             finished_ns=(folder/'status').stat().st_mtime_ns,transfer_bytes=archive.stat().st_size if archive.is_file() else None))
    out[seed]=scans
print(json.dumps(dict(time_ns=time.time_ns(),jobs=out)))'''
    for host in sorted({r['host'] for r in manifest}):
        group=[r for r in manifest if r['host']==host]
        query=code.replace('JOBS',repr([(r['seed'],r['policy_sha256']) for r in group]))
        start=time.time_ns()
        reply=subprocess.run([*fleet[host]['command'][:-1],'python3 -c '+shlex.quote(query)],stdin=subprocess.DEVNULL,capture_output=True,check=True,timeout=25)
        stop=time.time_ns();res=json.loads(reply.stdout)
        delta=res['time_ns']-(start+stop)/2
        clocks[host]=dict(estimated_offset_s=delta/1e9,roundtrip_s=(stop-start)/1e9)
        for r in group:
            d=cohort/f"{host}-{r['seed']}"
            with np.load(d/'behavior.npz') as b:action_start=int(b['action_utc_ms'][0])/1000
            with gzip.open(d/'native.csv.gz','rt') as f:terminal=int(list(csv.DictReader(f))[-1]['utc_ms'])/1000
            claims=res['jobs'][str(r['seed'])];ok=[c for c in claims if c['status']==0]
            assert len(ok)==1 and ok[0]['transfer_bytes']>0
            winning=ok[0];claim=winning['claimed_ns']/1e9;done=winning['finished_ns']/1e9
            all_credits.append(dict(host=host,seed=r['seed'],steps=r['steps'],startup_seconds=action_start-claim,
                active_seconds=terminal-action_start,teardown_and_compression_seconds=done-terminal,
                attempts=len(claims),bytes=winning['transfer_bytes'],first_dispatch_s=(min(c['claimed_ns'] for c in claims)-delta)/1e9,
                closed_s=(winning['finished_ns']-delta)/1e9,first_play_s=action_start-delta/1e9,last_play_s=terminal-delta/1e9))
    span=max(c['closed_s'] for c in all_credits)-min(c['first_dispatch_s'] for c in all_credits)
    active=sum(c['active_seconds'] for c in all_credits)
    events=sorted([(c['first_play_s'],1) for c in all_credits]+[(c['last_play_s'],-1) for c in all_credits])
    parallel=np.cumsum([d for t,d in events]);steps=sum(c['steps'] for c in all_credits)
    summary=dict(credits=len(all_credits),decisions=steps,total_dispatch_to_last_sealed_seconds=span,aggregate_decisions_per_second=steps/span,
        sum_of_verified_original_play_seconds=active,average_simultaneously_active_games=active/span,
        peak_simultaneously_active_games=int(parallel.max()),confirmed_bytes_downloaded_once=sum(c['bytes'] for c in all_credits),
        median_winning_attempt_setup_seconds=float(np.median([c['startup_seconds'] for c in all_credits])),
        median_winning_attempt_teardown_and_compression_seconds=float(np.median([c['teardown_and_compression_seconds'] for c in all_credits])),
        technical_attempts_before_winning=sum(c['attempts']-1 for c in all_credits))
    result=dict(scope='First remote claim to last verified remote original archive. Per-host clocks aligned to local request midpoint; uncertainty at most the stated SSH roundtrip. Includes remote setup, native play, retries, inter-job gaps and remote compression. Excludes initial model deployment and final learner-side rescoring/data packaging. The active game window is first actual game action through native final life loss.',host_time=clocks,summary=summary,credits=all_credits)
    (cohort/'fleet-throughput.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',type=Path,required=True);p.add_argument('--workers',type=Path,required=True)
    a=p.parse_args();audit(a.cohort,a.workers)
