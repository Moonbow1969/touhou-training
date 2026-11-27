"""Read current user-owned actor service memory and disk usage from the rollout hosts."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess


REMOTE = r"""
from datetime import datetime,timezone
from pathlib import Path
import json,os,shutil,subprocess
stats={}
for proc in Path('/proc').glob('[0-9]*'):
    try:
        args=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
        if len(args)<2 or not args[1].endswith('/camera_service.py'):continue
        group='th10' if 'th10-official-trial' in args[1] else 'other_camera_service'
        values={line.split()[0].strip(':'):int(line.split()[1]) for line in (proc/'smaps_rollup').read_text().splitlines()[1:] if len(line.split())==3 and line.split()[2]=='kB'}
        a=stats.setdefault(group,dict(processes=0,pss_kib=0,private_kib=0,rss_kib=0))
        a['processes']+=1;a['pss_kib']+=values['Pss'];a['rss_kib']+=values['Rss']
        a['private_kib']+=values['Private_Dirty']+values['Private_Clean']
    except (OSError,ValueError,KeyError):continue
memory={line.split()[0].strip(':'):int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split())==3}
base=Path.home()/'.local/share/toho-ml/th10-official-trial-002a'
volume={}
for label,path in [('original_game',base),('temporary',Path('/tmp'))]:
    total,used,free=shutil.disk_usage(path);volume[label]=dict(total_bytes=total,used_bytes=used,free_bytes=free)
spools={}
for key in ('full-story','full-models','camera-services'):
    folder=base/key
    if folder.exists():spools[key]=int(subprocess.check_output(['du','-s','-B1',str(folder)]).split()[0])
print(json.dumps(dict(observed_utc=datetime.now(timezone.utc).isoformat(),logical_cpu=os.cpu_count(),
    load_average=os.getloadavg(),available_memory_kib=memory['MemAvailable'],total_memory_kib=memory['MemTotal'],
    service_memory=stats,disk=volume,th10_allocated_bytes=spools)))
"""


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",required=True,type=Path);parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args();profiles=json.loads(args.workers.read_bytes())
    audit={}
    for host in ("momentum","accelerator"):
        cmd=profiles[host]["command"][:-1]+["python3 -c "+shlex.quote(REMOTE)]
        audit[host]=json.loads(subprocess.check_output(cmd,timeout=30))
    result=dict(scope="Snapshot of readable user-owned camera services. PSS apportions shared pages; RSS counts shared pages in each process. The inventory includes active and inactive checkpoint processes. Available memory and actual game-volume disk are shown separately from tmpfs.",hosts=audit)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")


if __name__=="__main__":
    main()
