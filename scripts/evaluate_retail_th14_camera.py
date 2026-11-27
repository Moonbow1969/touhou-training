"""Play an original six-stage TH14 retail Hard credit from the beginning with a frozen camera actor."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from scripts.benchmark_trial_workers import TrialWorker
from scripts.checkpoint_storage import digest
from scripts.train_distributed_native_ppo import action_mask


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('assets','clock','policy','out'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--rank-menu-tap',type=int,required=True,choices=range(1,13))
    p.add_argument('--image',default='toho-gamer-trial:v6')
    p.add_argument('--display-config',type=Path,help='Optional genuine local retail display setup, copied into the private profile.')
    p.add_argument('--windows-pid',type=int,help='Use an already prepared owned retail process on the native Windows desktop.')
    p.add_argument('--distro',default='Ubuntu-26.04')
    args=p.parse_args()
    sha=digest(args.assets/'th14.exe')
    if sha!='4609e701b677b8317e43be92041f0c7075a292182848a7c11215c3a226034f90':raise ValueError('Unexpected retail build')
    for asset in ('th14.dat','thbgm.dat'):
        if not (args.assets/asset).is_file():raise ValueError('Missing licensed retail data')
    out=args.out
    out.mkdir(parents=True,exist_ok=True)
    if (out/'credit.json').is_file():raise ValueError('Credit already sealed')
    docker=['wsl','-d',args.distro,'--exec','docker']
    def path(f):
        return subprocess.check_output(['wsl','-d',args.distro,'--exec','wslpath','-a','-u',f.resolve().as_posix()],text=True).strip()
    launcher=Path(__file__).resolve().parents[1]/'tools/rollout-worker-probe/run-th14-trial.sh'
    name=f'th14-retail-camera-{args.seed}'
    if args.windows_pid:
        import os
        import psutil
        import win32gui
        import win32process
        from PIL import ImageGrab
        hwnd=win32gui.GetForegroundWindow()
        if (win32process.GetWindowThreadProcessId(hwnd)[1]!=args.windows_pid
                or Path(psutil.Process(args.windows_pid).exe()).resolve()!= (args.assets/'th14.exe').resolve()
                or win32gui.GetClientRect(hwnd)!=(0,0,640,480)):
            raise ValueError('Prepared retail native foreground differs')
        os.environ['TH14_HARD_MENU_TAP']=str(args.rank_menu_tap)
        cmd=[str(args.clock.resolve())]
        def screenshot(filename):
            if win32gui.GetForegroundWindow()!=hwnd:raise ValueError('Lost owned retail window')
            ImageGrab.grab(window=hwnd).save(out/filename)
    else:
        cmd=[*docker,'run','--init','--rm','-i','--network','none','--name',name,'--cpus','2.5','--memory','1400m',
             '-e','TH14_RETAIL_CAMERA=1','-e',f'TH14_HARD_MENU_TAP={args.rank_menu_tap}','-e','TH14_SETTLED_CAPTURE=1']
        for src,dest,ro in ((args.assets,'/input/game',True),(args.clock,'/probe/trial-clock.exe',True),(launcher,'/probe/run-th14-retail.sh',True),(out,'/capture',False)):
            cmd+=['--mount',f'type=bind,source={path(src)},target={dest}'+(',readonly' if ro else '')]
        if args.display_config:
            cmd+=['--mount',f'type=bind,source={path(args.display_config)},target=/input/display.cfg,readonly']
        cmd+=['--entrypoint','sh',args.image,'/probe/run-th14-retail.sh','serve']
        def screenshot(filename):
            subprocess.run([*docker,'exec','-e','DISPLAY=:66',name,'scrot','-o','/capture/'+filename],check=True)
    model=torch.jit.load(str(args.policy),map_location='cpu').eval()
    torch.set_num_threads(1)
    rng=np.random.default_rng(args.seed)
    frames=[];actions=[];logp=[];states=[];stages=[];entry=None;prev=None
    start=time.monotonic()
    with TrialWorker(cmd,shutdown_timeout=15) as worker,torch.inference_mode():
        s=worker.read(timeout=140)
        if (s['difficulty'],s['stage'],s['lives'],s['bombs'],s['mode'],s['continues'])!=(2,1,2,3,2,0):
            raise ValueError('Original did not confirm Hard Reimu A full stock')
        screenshot('opening.png')
        while True:
            if s['continues'] or s['difficulty']!=2:raise ValueError('Rank changed or Continue observed')
            if not stages or s['stage']!=stages[-1]:stages.append(s['stage'])
            if s['stage']==2 and entry is None:entry={k:s[k] for k in ('stage','ticks','lives','bombs')}
            if s['ending']:
                if stages!=list(range(1,7)):raise ValueError('Ending without all six original stages')
                outcome='one_credit_clear';break
            if s['lives']<0 and s['mode']==1:outcome='game_over';break
            if time.monotonic()-start>5400:raise TimeoutError('Original credit did not reach a verified endpoint')
            if s['lives']<0:
                s=worker.request('S 0');continue
            raw=np.frombuffer(bytes.fromhex(s['camera']),dtype=np.uint8).reshape(112,96).copy()
            image=torch.from_numpy(raw).float()/255
            diff=torch.zeros_like(image) if prev is None else (image-prev).abs()
            probs=[h[0].softmax(0).numpy().astype(float) for h in model(torch.stack((image,diff))[None])[:4]]
            a=[int(rng.choice(len(q),p=q/q.sum())) for q in probs]
            frames.append(raw);actions.append(a);logp.append([np.log(q[k]) for q,k in zip(probs,a)])
            states.append([s[k] for k in ('ticks','frame','stage','lives','bombs','mode','ending','continues')])
            if len(actions)==1:print('ACTIVE original retail Hard, full stock',flush=True)
            prev=image;s=worker.request('S '+str(action_mask(a)))
        screenshot('endpoint.png')
    trace=out/'trajectory.npz'
    np.savez_compressed(trace,frames=np.stack(frames),actions=np.asarray(actions,np.int8),logp=np.asarray(logp,np.float32),native=np.asarray(states,np.int64))
    result=dict(build='TH14 retail 1.00b',original_sha256=sha,model_sha256=digest(args.policy),clock_sha256=digest(args.clock),
                difficulty='Hard',seed=args.seed,initial_spares=2,initial_bombs=3,stages=stages,stage2_entry=entry,
                terminal=outcome,terminal_spares=max(0,s['lives']),native_terminal_spares=s['lives'],terminal_bombs=s['bombs'],
                final_native_mode=s['mode'],continues=s['continues'],decisions=len(actions),wall_seconds=time.monotonic()-start,
                last_observed_stage1_tick=max(v[1] for v in states if v[2]==1),trace_sha256=digest(trace),
                initial_image_sha256=digest(out/'opening.png'),end_image_sha256=digest(out/'endpoint.png'))
    (out/'credit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
