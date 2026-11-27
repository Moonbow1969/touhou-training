"""Measure one sealed generation's acquisition, preparation and learner latency on the coordinator."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.checkpoint_storage import digest
from scripts.score_original_camera_run import records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    a=parser.parse_args()
    root=a.root.resolve()
    dest=root/'coordinator-timing.json'
    if dest.exists():
        raise FileExistsError('Coordinator timestamps already recorded')
    marks={}
    for key,relative in [('registered','camera-protocol.json'),('last_lane_qualified','th12-train-source-selection.json'),
                         ('camera_manifest','camera-batch/credits.json'),('learner_input','corrected/clock-audit.json'),
                         ('fitted','policy/report.json')]:
        path=root/relative
        marks[key]=dict(file=relative,sha256=digest(path),utc_ns=path.stat().st_mtime_ns,
                        utc=datetime.fromtimestamp(path.stat().st_mtime,tz=timezone.utc).isoformat())
    times=[m['utc_ns'] for m in marks.values()]
    if times!=sorted(times):
        raise ValueError('Recorded coordinator order is inconsistent')
    camera=json.loads((root/'camera-batch/credits.json').read_bytes())
    signed={(r['title'],r['seed']):r['source_log_sha256'] for r in camera}
    host_counts=Counter()
    sealed={}
    for title,f in records(root,True,True):
        r=json.loads(f.read_bytes())
        key=title,r.get('seed')
        if key not in signed:
            continue
        if digest(f)!=signed[key] or key in sealed:
            raise ValueError('Duplicate or unsigned original input')
        host=r['host'].lower().replace('steamdeck','deck')
        host_counts[host]+=1
        sealed[key]=dict(source=f.relative_to(root).as_posix(),sha256=signed[key],host=host,utc_ns=f.stat().st_mtime_ns)
    if set(sealed)!=set(signed):
        raise ValueError('Incomplete original acquisition')
    intervals={k:(v-u)/1e9 for k,(u,v) in zip(('collection_and_qualification','camera_packing','reward_preparation','learner_and_export'),zip(times,times[1:]))}
    result=dict(scope='Elapsed times from original Windows coordinator file seals during this live generation. Collection includes competing queues, deployment, verification and technical retries. Packing includes polling. Learner includes input re-forward, four PPO epochs and checkpoint export. These are end-to-end intervals for this run, not isolated hardware benchmarks or exclusive resource use.',
                coordinator_markers=marks,elapsed_seconds=intervals,total_seconds=(times[-1]-times[0])/1e9,
                accepted_training_credits_by_physical_host=dict(host_counts),
                original_receipts=[dict(title=t,seed=s,**r) for (t,s),r in sorted(sealed.items())])
    dest.write_text(json.dumps(result,indent=2)+'\n')
    print(intervals,dict(host_counts),flush=True)


if __name__=='__main__':
    main()
