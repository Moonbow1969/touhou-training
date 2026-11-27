"""Sample native game and inference cgroups while original Linux rollout actors run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

def group(container):
    pid=subprocess.check_output(['docker','inspect','--format','{{.State.Pid}}',container],text=True).strip()
    entry=next(line.split(':',2)[2] for line in (Path('/proc')/pid/'cgroup').read_text().splitlines() if line.startswith('0::'))
    root=Path('/sys/fs/cgroup')/entry.lstrip('/')
    if not (root/'cpu.stat').is_file():raise RuntimeError('A cgroup v2 host namespace is required')
    return root

def sample(root):
    cpu={k:int(v) for k,v in (line.split() for line in (root/'cpu.stat').read_text().splitlines())}
    live=[]
    for pid in (root/'cgroup.procs').read_text().split():
        try:live.append((Path('/proc')/pid/'comm').read_text().strip())
        except FileNotFoundError:pass
    return dict(usage_us=cpu['usage_usec'],periods=cpu['nr_periods'],throttled_periods=cpu['nr_throttled'],
                throttled_us=cpu['throttled_usec'],memory_bytes=int((root/'memory.current').read_text()),
                active_actor=any(x=='worker' for x in live))

def host_counters():
    fields=[int(v) for v in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
    mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    return dict(cpu_total_jiffies=sum(fields),cpu_idle_jiffies=fields[3]+fields[4],available_mib=int(mem['MemAvailable'].split()[0])/1024)

def metrics(entries,role):
    data=entries if role=='service' else [x for x in entries if x['actor_started_and_ended_active']]
    if not data:return dict(intervals=0)
    wall=sum(x['seconds'] for x in data);period=sum(x['periods'] for x in data)
    return dict(intervals=len(data),sampled_wall_s=wall,mean_cpu_cores=sum(x['cpu_seconds'] for x in data)/wall,
                max_interval_cpu_cores=max(x['cpu_seconds']/x['seconds'] for x in data),
                max_memory_mib=max(x['memory_bytes'] for x in data)/(1024**2),
                fraction_quota_periods_throttled=sum(x['throttled_periods'] for x in data)/period if period else None,
                throttled_cpu_seconds=sum(x['throttled_seconds'] for x in data))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--game',action='append',default=[]);parser.add_argument('--service',action='append',default=[])
    parser.add_argument('--samples',type=int,default=37);parser.add_argument('--period',type=float,default=5)
    parser.add_argument('--until-file',type=Path,help='Finish when this path exists after at least two samples')
    parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    if not a.game or a.samples<2 or a.period<=0:raise ValueError('No games or insufficient sample length')
    containers={c:(group(c),'game' if c in a.game else 'service') for c in [*a.game,*a.service]}
    plan={c:dict(role=role,quota=(root/'cpu.max').read_text().strip(),memory_limit_bytes=(root/'memory.max').read_text().strip()) for c,(root,role) in containers.items()}
    readings=[];start=time.monotonic()
    for n in range(a.samples):
        if n:time.sleep(max(0,start+n*a.period-time.monotonic()))
        readings.append(dict(monotonic=time.monotonic(),host=host_counters(),containers={c:sample(root) for c,(root,_) in containers.items()}))
        if n>=1 and a.until_file and a.until_file.exists():break
    intervals={c:[] for c in containers}
    for old,new in zip(readings,readings[1:]):
        duration=new['monotonic']-old['monotonic']
        for c in containers:
            before=old['containers'][c];after=new['containers'][c]
            intervals[c].append(dict(seconds=duration,cpu_seconds=(after['usage_us']-before['usage_us'])/1.e6,
                periods=after['periods']-before['periods'],throttled_periods=after['throttled_periods']-before['throttled_periods'],
                throttled_seconds=(after['throttled_us']-before['throttled_us'])/1.e6,memory_bytes=max(before['memory_bytes'],after['memory_bytes']),
                actor_started_and_ended_active=before['active_actor'] and after['active_actor']))
    result=dict(method='Linux cgroup-v2 cumulative counters. A game interval is summarized when the worker process is present at both sample endpoints; this can include brief intra-interval setup. Service intervals cover the whole observation window. CPU units are logical cores. WSL virtual CPU count is distinct from physical Windows host occupancy.',
                visible_logical_cpus=len(os.sched_getaffinity(0)),plan=plan,period_seconds=a.period,readings=readings,
                summary={c:metrics(items,containers[c][1]) for c,items in intervals.items()})
    duration=readings[-1]['monotonic']-readings[0]['monotonic']
    before,after=[r['host'] for r in (readings[0],readings[-1])]
    jiffies=after['cpu_total_jiffies']-before['cpu_total_jiffies']
    busy=1-(after['cpu_idle_jiffies']-before['cpu_idle_jiffies'])/jiffies
    used=sum((readings[-1]['containers'][c]['usage_us']-readings[0]['containers'][c]['usage_us'])/1e6/duration for c in containers)
    result['overall_host']=dict(mean_busy_logical_cores=busy*result['visible_logical_cpus'],observed_cgroups_mean_cores=used,
                               other_guest_mean_busy_cores=busy*result['visible_logical_cpus']-used,
                               minimum_available_mib=min(r['host']['available_mib'] for r in readings))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['summary'],indent=2))
if __name__=='__main__':main()
