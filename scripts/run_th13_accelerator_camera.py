"""Run original TH13 camera credits on the independently qualified Accelerator host."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import sys
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
    options=SimpleNamespace(host=target,host_label='accelerator',known_hosts=Path(known),
        remote_root=base+'/run',prior_bridge=base+'/toho-th13-cadence-261125/bridge',
        assets=base+'/toho-th13-261125/assets/app',launcher=base+'/toho-th13-lm-261125/src/launch.sh',
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
    run(p.parse_args())
