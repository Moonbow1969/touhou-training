"""Measure Wine prefix initialization separately from original gameplay, using one disposable container."""
import hashlib
import argparse
import json
from pathlib import Path
import shlex
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--workers',type=Path,required=True);p.add_argument('--host',required=True);p.add_argument('--out',type=Path,required=True)
args=p.parse_args()
cfg=json.loads(args.workers.read_bytes())[args.host]
body=r'''set -eu
export DISPLAY=:59 WINEDEBUG=-all WINEDLLOVERRIDES=mscoree,mshtml= LANG=C.UTF-8
Xvfb "$DISPLAY" -screen 0 1024x768x24 -nolisten tcp >/tmp/x.log 2>&1 &
display_pid=$!
trap 'kill "$display_pid" 2>/dev/null || true' EXIT
for repeat in 1 2 3; do
  export WINEPREFIX="/tmp/th10-private-prefix-probe-$repeat"
  for mode in cold reused; do
    t0=$(date +%s%N)
    if timeout 120 wineboot --init >"/tmp/boot-$repeat-$mode.log" 2>&1; then rc=0; else rc=$?; fi
    t1=$(date +%s%N)
    wineserver -k >/dev/null 2>&1 || true
    timeout 15 wineserver -w >/dev/null 2>&1 || true
    registered=0; if test -s "$WINEPREFIX/system.reg" && test -s "$WINEPREFIX/user.reg"; then registered=1; fi
    echo "$repeat,$mode,$rc,$((t1-t0)),$registered"
    if test "$rc" != 0 || test "$registered" != 1; then ls -l "$WINEPREFIX" >&2; cat "/tmp/boot-$repeat-$mode.log" >&2; exit 2; fi
  done
done'''
tag=cfg['runtime']
cmd='docker run --rm --init --network=none --cpus=2 --memory=1536m --log-driver=local --log-opt=max-size=1m --entrypoint /bin/sh '+shlex.quote(tag)+' -c '+shlex.quote(body)
r=subprocess.run([*cfg['command'][:-1],cmd],stdin=subprocess.DEVNULL,capture_output=True,timeout=600)
if r.returncode:raise RuntimeError((r.stdout+r.stderr).decode(errors='replace')[-1000:])
rows=[]
for line in r.stdout.decode().splitlines():
    rep,mode,code,elapsed,reg=line.split(',')
    rows.append(dict(replicate=int(rep),mode=mode,exit_code=int(code),seconds=int(elapsed)/1e9,default_registry_verified=bool(int(reg))))
assert len(rows)==6
medians={mode:sorted(x['seconds'] for x in rows if x['mode']==mode)[1] for mode in ('cold','reused')}
result=dict(host=args.host,container_image=tag,game_mounted=False,network_enabled=False,retained_container_or_cache=False,
    normalized_scope='One disposable original-game runtime and one existing Xvfb. Three distinct empty Wine prefixes. Within each pair compare wineboot --init on empty prefix vs same initialized default prefix after stopping its wineserver. No game is installed or launched; captured times cover Wine prefix initialization only.',
    median_seconds=medians,receipts=rows,script_sha256=hashlib.sha256(body.encode()).hexdigest())
args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
