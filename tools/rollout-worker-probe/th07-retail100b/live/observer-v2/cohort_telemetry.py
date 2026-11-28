"""Summarize original credits, control timing and physical placement for a frozen retail cohort."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def summarize(folder):
    cohort=json.loads((folder/"completed-pairs.json").read_bytes())
    records=[]
    actor_intervals={}
    for pair in cohort["pairs"]:
        for arm in pair["order"]:
            case=folder/f"{pair['seed']}-{arm}"
            q=pair["arms"][arm]
            row=dict(seed=pair["seed"],arm=arm,eligible_pair=pair["valid"],eligible_credit=q["valid"],slot=pair["slot"])
            if (case/"verified.json").is_file():
                r=json.loads((case/"verified.json").read_bytes())
                for k in ("stage1_native_ticks","stage1_deaths","stage2","stage2_resources","terminal_stock","actual_bombs",
                          "score_proof","observer_p999_upper_ms","max_synchronous_observer_ms","max_freeze_ms","native_wall_seconds",
                          "exact_fraction","complete_windows","strict_exact","skipped_decisions","stale","partial","terminal_continues"):
                    row[k]=r[k]
                row["sealed_records"]=r["archive"]["records"]
                row["worst_read_end_ns"]=int(r["read_distribution"]["max_end_ns"])
                row["verified_sha256"]=hashlib.sha256((case/"verified.json").read_bytes()).hexdigest()
            if (case/"cpu-binding.json").is_file() and (case/"cgroup.jsonl").is_file():
                binding=json.loads((case/"cpu-binding.json").read_bytes())
                cpus={int(c) for c in binding["physical_cpu_ids"].split(",")}
                samples=[json.loads(line) for line in (case/"cgroup.jsonl").read_bytes().splitlines()]
                stats=[dict(line.split() for line in samples[j]["cpu"].splitlines()) for j in (0,-1)]
                delta=samples[-1]["ns"]-samples[0]["ns"]
                actor_intervals[case.name]=(samples[0]["ns"],samples[-1]["ns"])
                observed=[cpu for s in samples for cpu in s["vm_busy_thread_cpus"] if cpu is not None]
                row.update(cpu_ids=sorted(cpus),cpu_max=binding["game_cgroup_cpu_max"],
                    game_container_mean_cpu_cores=(int(stats[1]["usage_usec"])-int(stats[0]["usage_usec"]))*1000/delta if delta else None,
                    cgroup_throttled_periods=int(stats[1]["nr_throttled"])-int(stats[0]["nr_throttled"]),
                    vm_worker_last_cpu_samples=len(observed),vm_worker_last_cpu_in_game_cpuset=sum(c in cpus for c in observed))
            records.append(row)
    for row in records:
        if row.get("max_synchronous_observer_ms",0)<100:
            continue
        t=row["worst_read_end_ns"]
        name=f"{row['seed']}-{row['arm']}"
        others={k:v for k,v in actor_intervals.items() if k!=name}
        boundaries=[(abs(t-v),k,label,(t-v)/1e9) for k,(a,b) in others.items()
                    for label,v in (("host-actor-start",a),("host-actor-end",b))]
        row["other_host_actors_active_at_longest_read"]=[k for k,(a,b) in others.items() if a<=t<=b]
        row["nearest_other_actor_boundaries"]=[dict(credit=k,boundary=label,seconds_after=v)
                                               for _,k,label,v in sorted(boundaries)[:2]]
    report=dict(scope="Each credit is a natural full original stock Game Start with an independently initialized game RNG. VM samples are the last CPU reported at each poll and need not indicate simultaneous use.",rows=records)
    arms=list(json.loads((folder/"held-protocol.json").read_bytes())["models"])
    eligible=[[r for r in records if r["eligible_pair"] and r["arm"]==arm] for arm in arms]
    if all(len(a)>=2 for a in eligible):
        rng=np.random.default_rng(2173751)
        rate=[]; mean=[]
        for a in eligible:
            d=np.array([r["stage1_deaths"] for r in a])
            t=np.array([r["stage1_native_ticks"] for r in a])
            ids=rng.integers(0,len(a),size=(20000,len(a)))
            rate.append(d[ids].sum(axis=1)/t[ids].sum(axis=1))
            mean.append(t[ids].mean(axis=1))
        keep=rate[0]>0
        report.update(exploratory_bootstrap="20,000 independent resamples of whole eligible credits within each arm. Percentile intervals describe the finite roster and do not alter the registered decision.",
                      stage1_rate_ratio_bootstrap_95=np.quantile(rate[1][keep]/rate[0][keep],[.025,.975]).tolist(),
                      stage1_mean_tick_delta_bootstrap_95=np.quantile(mean[1]-mean[0],[.025,.975]).tolist())
    return report


if __name__=="__main__":
    root=Path(sys.argv[1])
    (root/"cohort-telemetry.json").write_text(json.dumps(summarize(root),indent=2)+"\n")
