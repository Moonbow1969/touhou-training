"""Run a frozen actor on its registered CPU group and retain game/VM scheduling telemetry."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def cgroup(pid):
    relative = Path(f"/proc/{pid}/cgroup").read_text().strip().split(":",2)[2]
    return Path("/sys/fs/cgroup")/relative.lstrip("/")


def vm_threads():
    # Read-only. Only the two highest cumulative-CPU QEMU worker threads are tracked.
    q=subprocess.run(["docker","inspect","--format","{{.State.Pid}}","moonbow-th08-qemu-original-261127"],
                     capture_output=True,text=True)
    if q.returncode:
        return []
    candidates=[]
    for ident in (cgroup(int(q.stdout))/"cgroup.procs").read_text().splitlines():
        base=Path("/proc")/ident
        try:
            if not (base/"comm").read_text().startswith("qemu-system"):
                continue
            for f in (base/"task").glob("*/stat"):
                fields=f.read_text().rsplit(")",1)[1].split()
                if f.parent.name != ident:
                    candidates.append((int(fields[11])+int(fields[12]),f))
        except FileNotFoundError:
            continue
    return [f for _,f in sorted(candidates,reverse=True)[:2]]


def run(folder, cpus, seed, arm):
    name=(folder/"actual-container").read_text().strip()
    info=json.loads(subprocess.check_output(["docker","inspect",name]))[0]
    group=cgroup(info["State"]["Pid"])
    assert info["HostConfig"]["CpusetCpus"]==cpus
    assert (group/"cpu.max").read_text().split()[0]=="max"
    threads=vm_threads()
    (folder/"cpu-binding.json").write_text(json.dumps(dict(physical_cpu_ids=cpus,
        game_cgroup_cpu_max=(group/"cpu.max").read_text().strip(),
        vm_thread_stat_files=[str(f) for f in threads],host=os.uname().nodename,
        measurement="VM observed scheduling CPU per live busy thread, every quarter second. The VM permits kernel migration."),indent=2)+"\n")
    cmd=["taskset","-c",cpus,sys.executable,"-u",str(Path(__file__).parent/"play_record.py"),seed,arm]
    process=subprocess.Popen(cmd)
    with (folder/"cgroup.jsonl").open("x",buffering=1) as f:
        while True:
            cpu=[]
            for path in threads:
                try:
                    cpu.append(int(path.read_text().rsplit(")",1)[1].split()[36]))
                except FileNotFoundError:
                    cpu.append(None)
            print(json.dumps(dict(ns=time.monotonic_ns(),cpu=(group/"cpu.stat").read_text(),
                cpu_pressure=(group/"cpu.pressure").read_text(),memory=(group/"memory.events").read_text(),
                vm_busy_thread_cpus=cpu)),file=f)
            if process.poll() is not None:
                return process.returncode
            time.sleep(.25)


if __name__=="__main__":
    sys.exit(run(Path(sys.argv[1]),*sys.argv[2:]))
