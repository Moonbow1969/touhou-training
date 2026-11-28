"""Join the native observer's largest calls to separately clocked host and core measurements."""
import csv,gzip,hashlib,json,re,shutil,sys
from pathlib import Path
root=Path(sys.argv[1])
def getcsv(p):
    with p.open() as f:return [{k:int(v) for k,v in row.items()} for row in csv.DictReader(f)]
def metric(s):return dict(line.split()[:2] for line in s.splitlines())
def total_psi(s,cls):
    return int(next(q.split("total=")[1] for q in s.splitlines() if q.startswith(cls+" ")))
def device(data):
    result={}
    for row in data.splitlines():
        a=row.split()
        if re.fullmatch(r"(sd[a-z]+|nvme\d+n\d+|vd[a-z]+)",a[2]):result[a[2]]=(int(a[5]),int(a[9]))
    return result
def resident(p,t):
    return max(i for i,x in enumerate(p) if x["ns"]<=t)
can=getcsv(root/"canary.csv")
host=[json.loads(x) for x in (root/"host.jsonl").read_text().splitlines()]
out=[]
for case in sorted(root.glob("217*-parent")):
    v=json.loads((case/"verified.json").read_bytes())
    cgroup=[json.loads(x) for x in (case/"cgroup.jsonl").read_text().splitlines()]
    t=int(v["read_distribution"]["max_end_ns"])
    first,last=cgroup[0]["ns"],cgroup[-1]["ns"]
    cr=[q for q in can if first<=q["sample_end_ns"]<=last]
    hw=[q for q in host if first<=q["ns"]<=last]
    c=next(q for q in can if q["sample_end_ns"]>=t)
    i=resident(host,t);h0,h1=host[i:i+2];d0,d1=h0["data"],h1["data"]
    i=resident(cgroup,t);g0,g1=cgroup[i:i+2]
    disks1,disks2=device(d0["/proc/diskstats"]),device(d1["/proc/diskstats"])
    span=(h1["ns"]-h0["ns"])/1e9
    peak=dict(observer_end_ns=t,observer_max_ms=int(v["read_distribution"]["max_ns"])/1e6,
            canary_bucket=c,canary_peak_minus_observer_end_ms=(c["peak_ns"]-t)/1e6,
            cgroup_span_ms=(g1["ns"]-g0["ns"])/1e6,
            cgroup_cpu_full_pressure_delta_ms=(total_psi(g1["cpu_pressure"],"full")-total_psi(g0["cpu_pressure"],"full"))/1000,
            cgroup_cpu_some_pressure_delta_ms=(total_psi(g1["cpu_pressure"],"some")-total_psi(g0["cpu_pressure"],"some"))/1000,
            host_span_s=span,host_io_full_pressure_delta_ms=(total_psi(d1["/proc/pressure/io"],"full")-total_psi(d0["/proc/pressure/io"],"full"))/1000,
            block_devices_mib_write_per_second={k:round((disks2[k][1]-disks1[k][1])/2048/span,4) for k in disks1.keys()&disks2.keys()})
    out.append(dict(seed=v["seed"],passed=v["passed"],canary_samples=sum(q["samples"] for q in cr),
                    max_canary_lateness_ms=max(q["max_late_ns"] for q in cr)/1e6,
                    canary_skipped_deadlines=sum(q["missed_wakes"] for q in cr),
                    host_max_io_some_avg10_pct=max(float(q["data"]["/proc/pressure/io"].split("avg10=")[1].split()[0]) for q in hw),
                    cgroup_throttled=[metric(q["cpu"])["nr_throttled"] for q in (cgroup[0],cgroup[-1])],
                    cgroup_oom=[metric(q["memory"])["oom_kill"] for q in (cgroup[0],cgroup[-1])],
                    cgroup_swap=[q["swap"].strip() for q in (cgroup[0],cgroup[-1])],peak=peak))
(root/"host-timing.json").write_text(json.dumps(dict(cases=out,resolution="1ms canary deadlines aggregated into one-second buckets; original per-call maximum and monotonic bracket exact; host reported each second. A maximum from the same bucket alone does not prove a shared cause."),indent=2)+"\n")
for name in ("canary.csv","host.jsonl"):
    with (root/name).open("rb") as src,gzip.open(root/(name+".gz"),"wb") as dst:shutil.copyfileobj(src,dst)
