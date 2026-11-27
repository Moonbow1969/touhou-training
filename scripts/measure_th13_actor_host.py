"""Measure CPU, bytes and free space on the Linux original-game worker."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def memory():
    return {line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
            if line.startswith(('MemAvailable:','MemTotal:'))}


def sample(path):
    # guest and guest_nice are already part of user/nice in Linux /proc/stat.
    raw=[int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
    return dict(monotonic_ns=time.monotonic_ns(),cpu_total_jiffies=sum(raw),cpu_busy_jiffies=sum(raw)-raw[3]-raw[4],
                hz=os.sysconf('SC_CLK_TCK'),cpus=os.cpu_count(),free_bytes=shutil.disk_usage(path).free,**memory())


def report(name,path):
    state=subprocess.check_output(['docker','inspect','-f','{{.State.Pid}}',name],text=True).strip()
    group=Path('/sys/fs/cgroup')/next(row.split('::',1)[1].lstrip('/') for row in Path(f'/proc/{state}/cgroup').read_text().splitlines() if '::' in row)
    cpu={key:int(v) for key,v in (row.split() for row in (group/'cpu.stat').read_text().splitlines())}
    io={'rbytes':0,'wbytes':0}
    if (group/'io.stat').is_file():
        for row in (group/'io.stat').read_text().splitlines():
            for item in row.split()[1:]:
                k,v=item.split('=')
                if k in io:io[k]+=int(v)
    sizes={k:int(subprocess.check_output(['du','-s',flag,str(path)]).split()[0]) for k,flag in
           [('disk_bytes','-B1'),('logical_bytes','-b')]}
    return dict(game_cgroup_cpu_usec=cpu['usage_usec'],cgroup_io=io,
        cgroup_memory_peak=(int((group/'memory.peak').read_text()) if (group/'memory.peak').exists() else None),**sizes)


if __name__=='__main__':
    print(json.dumps(sample(sys.argv[2]) if sys.argv[1]=='snapshot' else report(sys.argv[2],Path(sys.argv[3]))))
