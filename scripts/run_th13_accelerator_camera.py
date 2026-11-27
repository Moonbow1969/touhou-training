"""Run original TH13 camera credits on the independently qualified Accelerator host."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace

NATIVE = Path(__file__).resolve().parents[1] / 'tools/rollout-worker-probe/th13-full'
sys.path.insert(0, str(NATIVE))
from run_camera_credit import CameraPool


def run(args):
    profile=json.loads((args.qualification / 'deployment.json').read_bytes())
    evidence=json.loads((args.qualification / 'qualification.json').read_bytes())
    if profile['target_host']!='accelerator' or not evidence['qualified'] or evidence['game_sha256']!=profile['asset_sha256']['toho-th13-261125/assets/app/th13.exe']:
        raise ValueError('Missing qualified TH13 original-capture runtime')
    if len(set(args.seeds))!=len(args.seeds) or any(s<0 for s in args.seeds):
        raise ValueError('Controller seeds must be unique nonnegative integers')
    if not args.arm or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.arm):
        raise ValueError('Use lowercase alphanumeric identifiers and dashes in arm names')
    conn=json.loads(args.connection_config.read_bytes())['accelerator']['command'][:-1]
    remote=lambda line:subprocess.check_output([*conn,line],timeout=30,text=True).strip()
    base=profile['private_remote_root'];image=profile['local_image']
    if remote('docker image inspect --format "{{.Id}}" '+shlex.quote(image))!=profile['local_image_id']:
        raise ValueError('Qualified Wine runtime image changed')
    if int(remote("df -Pk /tmp | tail -1 | awk '{print $4}'"))<8*1024**2:
        raise OSError('The original rollout volume is below its 8 GiB operating reserve.')
    files=profile['asset_sha256']
    sums=remote('cd '+shlex.quote(base)+' && sha256sum '+shlex.join(files)).splitlines()
    if {row.split()[1]:row.split()[0] for row in sums}!=files:
        raise ValueError('Qualified original assets, launcher or observer changed')
    target=conn[-1];known=next(x.split('=',1)[1] for x in conn if x.startswith('UserKnownHostsFile='))
    python=remote('v="$HOME/.local/share/toho-ml/distributed-ppo/venv/bin/python"; "$v" -c "import torch" && printf "%s" "$v"')
    rank=getattr(args,'difficulty',1)
    if rank not in (1,2,3):raise ValueError('Supported native story ranks are Normal, Hard and Lunatic')
    launcher=base+'/toho-th13-lm-261125/src/launch.sh';runtime=base+'/run'
    if rank>1:
        source=(NATIVE.parent/'th13-pixels/launch.sh').read_bytes().replace(b'\r\n',b'\n')
        fingerprint=hashlib.sha256(source).hexdigest()
        runtime=base+'/ranked-story-'+fingerprint[:16]+'-'+str(rank)
        launcher=runtime+'/launch.sh'
        upload=('set -eu; umask 077; mkdir -p '+shlex.quote(runtime)+'; '
            't=$(mktemp '+shlex.quote(runtime+'/launcher.XXXXXXXX')+'); cat >"$t"; '
            'set -- $(sha256sum "$t"); test "$1" = '+fingerprint+'; mv "$t" '+shlex.quote(launcher))
        subprocess.run([*conn,upload],input=source,check=True,timeout=25)
        if remote('sha256sum '+shlex.quote(launcher)).split()[0]!=fingerprint:
            raise ValueError('Isolated rank launcher does not match local source')
        prefix='toho-th13-lm-261125/src/'
        for original in sorted(f for f in files if f.startswith(prefix) and not f.endswith('/launch.sh')):
            filename=original[len(prefix):]
            target_file=runtime+'/'+filename
            remote('cp -T '+shlex.quote(base+'/'+original)+' '+shlex.quote(target_file))
            if remote('sha256sum '+shlex.quote(target_file)).split()[0]!=files[original]:
                raise ValueError('Isolated rank companion does not match qualified runtime')
        remote('gcc -std=c99 -Wall -Wextra -Werror -fsyntax-only '+shlex.quote(runtime+'/pixel-capture.c'))
        args.output.mkdir(exist_ok=True,parents=True)
        staging=tempfile.NamedTemporaryFile(dir=args.output,mode='w',encoding='utf-8',delete=False)
        with staging:
            json.dump(dict(game_sha256=evidence['game_sha256'],launch_sha256=fingerprint,
                companion_sha256={f[len(prefix):]:files[f] for f in files if f.startswith(prefix) and not f.endswith('/launch.sh')},
                upstream='tools/rollout-worker-probe/th13-pixels/launch.sh',supported_ranks=[2,3],
                qualification='Isolated original launch with TH13_RANK; stages/characters from native official observer.',
                native_rank_column=141),staging,indent=2);staging.write('\n')
        Path(staging.name).replace(args.output/'rank-launcher.json')
    options=SimpleNamespace(host=target,host_label='accelerator',known_hosts=Path(known),
        remote_root=runtime,prior_bridge=base+'/toho-th13-cadence-261125/bridge',
        assets=base+'/toho-th13-261125/assets/app',launcher=launcher,difficulty=rank,
        image=image,python_executable=python,checkpoint=args.checkpoint.resolve(),output=args.output.resolve(),
        role=args.role,arm=args.arm)
    pool=CameraPool(options)
    pool.prepare()
    with ThreadPoolExecutor(args.workers) as workers:list(workers.map(pool.run_credit,args.seeds))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('qualification','connection-config','checkpoint','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--arm',required=True);p.add_argument('--role',choices=('train','held'),required=True)
    p.add_argument('--seeds',type=int,nargs='+',required=True);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--difficulty',type=int,choices=(1,2,3),default=1)
    run(p.parse_args())
