"""Measure local TH10 camera inference and the post-credit network transfer, excluding setup."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',type=Path,required=True)
    p.add_argument('--workers',type=Path,required=True)
    args=p.parse_args()
    cohort=args.cohort.resolve()
    rows=[json.loads(f.read_bytes()) for f in sorted(cohort.glob('*/report.json'))]
    if not rows or not all(r['terminal'] and r['camera_policy'] for r in rows):
        raise ValueError("Incomplete or absent native camera credits")
    assert len({(r['host'],r['policy_sha256'],r['source']['bundle']) for r in rows})==1
    host,sha,bundle=rows[0]['host'],rows[0]['policy_sha256'],rows[0]['source']['bundle']
    assert re.fullmatch('[0-9a-f]{64}',sha) and re.fullmatch('[0-9a-f]{16}',bundle)
    remote=json.loads(args.workers.read_bytes())[host]['command'][:-1]
    def fetch(script):
        return subprocess.run([*remote,script],capture_output=True,check=True,stdin=subprocess.DEVNULL,timeout=30).stdout.decode()
    root='"$HOME/.local/share/toho-ml/th10-official-trial-002a"'
    log=fetch('cd '+root+" && cat "+shlex.quote(f'camera-services/{sha}-{bundle}/service.log'))
    events=[json.loads(line) for line in log.splitlines() if line.startswith('{"decisions"')]
    # One server can serve the distinct train and held cohorts of the same checkpoint concurrently.
    # A step count uniquely identifies a credit only when that count occurs in exactly one service event.
    service=[]
    for row in rows:
        match=[event for event in events if event['decisions']==row['steps']]
        if len(match)==1 and sum(r['steps']==row['steps'] for r in rows)==1:
            service.append(dict(seed=row['seed'],**match[0]))
    commands=['set -eu','cd '+root]
    for row in rows:
        seed=int(row['seed'])
        hold=int(row.get('requested_hold_ms',120))
        label=f'{seed}-{sha[:12]}'+(f'-h{hold}' if hold!=120 else '')
        commands.append(f'for attempt in 1 2 3; do file="full-story/{label}-$attempt/behavior.tar.gz"; '
                        f'if test -f "$file" && test "$(cat "full-story/{label}-$attempt/status")" = 0; then '
                        f'echo {seed} $(wc -c <"$file"); break; fi; done')
    transfers={int(seed):int(size) for seed,size in (s.split() for s in fetch('\n'.join(commands)).splitlines())}
    assert set(transfers)=={r['seed'] for r in rows}
    intervals=[]
    for r in rows:
        d=cohort/f"{host}-{r['seed']}"
        with np.load(d/'original-pixels.npz') as camera:
            intervals.extend(np.diff(camera['monotonic_ns']).tolist())
    all_ms=np.array(intervals)/1e6
    service_total=sum(s['decisions'] for s in service)
    steps=sum(r['steps'] for r in rows)
    history=int(bool(rows[0]['source'].get('camera_history')))
    out=dict(credits=len(rows),policy_sha256=sha,host=host,decisions=steps,nominal_hold_ms=rows[0].get('requested_hold_ms',120),
             observed_camera_interval_ms={'median':float(np.median(all_ms)),'p95':float(np.quantile(all_ms,.95))},
             service_wall_ms_weighted_mean=(sum(m['mean_service_ms']*m['decisions'] for m in service)/service_total if service_total else None),
             service_coverage=dict(uniquely_matched_credits=len(service),decisions=service_total,
                                  note='Per-process service logs are joined by exact unique decision count; ambiguous counts are excluded.'),
             service_scope='CPU server time after receiving the frame through sending the response; private Unix socket. No LAN request per decision.',
             service_samples=service,per_decision_camera_payload_bytes=112*96,per_decision_reply_bytes=16*4,
             previous_move_bytes_per_step=history,total_local_socket_payload_bytes=steps*(112*96+history+64)+(64+2*history)*len(rows),
             trajectory_download_bytes=sum(transfers.values()),download_bytes_per_decision=sum(transfers.values())/steps,
             transferred_archives_by_seed=transfers,
             transfer_scope='One gzip output archive per credit, downloaded after native termination. SSH control traffic, TCP framing, retries and initial host setup excluded.')
    file=cohort/'runtime-cost.json'; file.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('service_samples','transferred_archives_by_seed')},indent=2))


if __name__=='__main__':
    main()
