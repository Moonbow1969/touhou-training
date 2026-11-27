"""Run one complete stock TH08 trial credit in private Wine/Xvfb using a frozen visual actor."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image
import torch

from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze


def linux(path):
    path=Path(path).resolve()
    return '/mnt/'+path.drive[0].lower()+path.as_posix()[2:]


def mask_for(action):
    move,focus,shot,bomb=action
    x=move%3-1
    y=move//3-1
    return int((x<0) | ((x>0)<<1) | ((y<0)<<2) | ((y>0)<<3) | (focus<<4) | (shot<<5) | (bomb<<6))


def area_camera(rgb):
    field=rgb[16:464,32:416].astype(np.uint32)
    grey=77*field[:,:,0]+150*field[:,:,1]+29*field[:,:,2]
    return ((grey.reshape(112,4,96,4).sum((1,3))+2048)//4096).astype(np.uint8)


def read_exact(handle,n):
    data=bytearray()
    while len(data)<n:
        chunk=handle.read(n-len(data))
        if not chunk:
            raise EOFError('Original-game pixel worker stopped before complete image')
        data.extend(chunk)
    return data


def stocks(rgb):
    rgb=rgb.astype(np.int16)
    result=[]
    for row,kind in ((96,'life'),(112,'bomb')):
        occupied=[]
        for x in range(496,609,16):
            r,g,b=np.moveaxis(rgb[row-6:row+7,x-6:x+7],2,0)
            mask=((r>125)&(r-g>32)&(r-b>36)) if kind=='life' else ((b>110)&(b-r>20)&(b-g>28))
            occupied.append(mask.sum()>=16)
        n=sum(occupied)
        result.append(int(n) if occupied==[True]*n+[False]*(len(occupied)-n) else None)
    return result


def text_structure(gray,edge,mask):
    patch=gray[173:207,110:323].astype(np.float32)
    contrast=(patch-cv2.GaussianBlur(patch,(0,0),1.35))[mask]
    reference=edge[mask]
    return float(np.dot(contrast,reference)/max(np.linalg.norm(contrast)*np.linalg.norm(reference),1e-6))


def run(args):
    tool=Path(__file__).resolve().parents[1]/'tools/rollout-worker-probe/th08-trial'
    root=args.out.resolve()
    root.mkdir(parents=True,exist_ok=True)
    game=args.assets/'game'
    src=json.loads((args.assets/'archived-official-trial.json').read_bytes())
    for name in ('th08tr.exe','th08tr.dat','th08.cfg'):
        if digest(game/name)!=src['trial_files'][name]:
            raise ValueError('Original demo files changed')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    actor=torch.jit.load(str(args.policy),map_location='cpu').eval()
    ending=np.load(tool/'continue-text-mask.npy',allow_pickle=False)
    with np.load(tool/'continue-edges.npz',allow_pickle=False) as edge_data:
        edge,edge_mask=edge_data['edge'],edge_data['mask']
    hud=np.load(tool/'game-hud-mask.npy',allow_pickle=False)
    plan=dict(title=8,rank=args.rank,seed=args.seed,role=args.role,
              policy=digest(args.policy),source=src['trial_files'],executable=digest(args.assets/'stream'),
              continue_template=digest(tool/'continue-text-mask.npy'),hud_template=digest(tool/'game-hud-mask.npy'),structural_continue_template=digest(tool/'continue-edges.npz'),
              field_xywh=[32,16,384,448],controls='All four heads sampled once per observation. Start when the original score HUD is visible. Stop at the first native continue prompt. No continue, fixed shot, geometric controller or emergency Bomb.',
              cadence='85 ms minimum after confirmed X keyboard application plus native capture, local IPC and CPU inference. Original game runs continuously.',
              timeout_seconds=480)
    if args.verified_menu:
        plan['observer_revision']='v2 independent structural continue check and original score-card or absent-display technical stop'
        plan['runner_sha256']=digest(Path(__file__))
    freeze(root/'protocol.json',plan)
    name=f'toho-th08-{args.rank.lower()}-{args.seed}'
    cmd=['wsl','-d','Ubuntu-26.04','--','docker','run','--rm','-i','--name',name,'--network','none','--cpus=1.70','--memory=1350m','--log-driver','none']
    for source,dest,ro in ((game,'/input/game',True),(args.assets/'stream','/input/stream',True),
                           (tool/'launch.sh','/input/run.sh',True),(root,'/capture',False)):
        cmd+=['--mount',f'type=bind,source={linux(source)},target={dest}'+(',readonly' if ro else '')]
    cmd+=['--entrypoint','sh','toho-gamer-trial:v6','/input/run.sh',args.rank]
    logs=[]
    hud_evidence=[];prompt_evidence=[]
    frames=[];actions=[];prob=[];values=[]
    movie=cv2.VideoWriter(str(root/'original-observations.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),8.,(640,480))
    if not movie.isOpened():
        raise RuntimeError('Movie encoder unavailable')
    out='unverified';prev=None;first_ns=None;began=False
    lost_since=None
    with (root/'container.stderr').open('wb') as err, subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err) as proc:
        try:
            while True:
                line=proc.stdout.readline()
                if not line.startswith(b'RGB '):
                    raise ValueError('Unexpected camera transport response '+repr(line[:100]))
                when=int(line.split()[1])
                if first_ns is None:first_ns=when
                rgb=np.frombuffer(read_exact(proc.stdout,640*480*3),np.uint8).reshape(480,640,3)
                mono=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
                hud_evidence.append(rgb[22:226,430:636].copy())
                prompt_evidence.append(mono[170:210,108:326].copy())
                hud_fraction=float(np.mean(mono[25:124,434:497][hud]>150))
                terminal_fraction=float(np.mean(mono[174:206,112:324][ending]>150))
                structure=text_structure(mono,edge,edge_mask)
                lives,bombs=stocks(rgb)
                began=began or hud_fraction>=.95
                movie.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                prompt=began and lives==0 and terminal_fraction>=.85 and structure>=.78
                if prompt and (args.verified_menu or hud_fraction>=.95):
                    out='original_continue_prompt'
                    Image.fromarray(rgb).save(root/'terminal.png')
                    logs.append(dict(frame=len(logs),monotonic_ns=when,hud=hud_fraction,continue_text=terminal_fraction,continue_structure=structure,spares=lives,bombs=bombs,phase='terminal'))
                    break
                if args.verified_menu and began:
                    if hud_fraction<.5:
                        lost_since=when if lost_since is None else lost_since
                    else:
                        lost_since=None
                    if lost_since is not None and when-lost_since>6_000_000_000:
                        out='technical_lost_game_hud'
                        Image.fromarray(rgb).save(root/'technical-stop.png')
                        logs.append(dict(frame=len(logs),monotonic_ns=when,hud=hud_fraction,
                                         continue_text=terminal_fraction,continue_structure=structure,
                                         spares=lives,bombs=bombs,phase='technical'))
                        break
                if (when-first_ns)*1e-9>plan['timeout_seconds']:
                    out='unverified_timeout'
                    Image.fromarray(rgb).save(root/'timeout.png')
                    break
                key=0
                row=dict(frame=len(logs),monotonic_ns=when,hud=hud_fraction,continue_text=terminal_fraction,
                         continue_structure=structure,spares=lives,bombs=bombs,phase='policy' if began else 'transition')
                if began:
                    camera=area_camera(rgb)
                    if prev is None:
                        Image.fromarray(rgb).save(root/'first-policy-frame.png')
                        prev=camera
                    inputs=np.stack([camera,np.abs(camera.astype(np.int16)-prev).astype(np.uint8)])[None]
                    with torch.inference_mode():
                        *heads,critic=actor(torch.from_numpy(inputs).float()/255.)
                        act=[int(torch.multinomial(logits.softmax(1),1)) for logits in heads]
                        lp=[float(logits.log_softmax(1)[0,a]) for logits,a in zip(heads,act)]
                    key=mask_for(act)
                    frames.append(camera);actions.append(act);prob.append(lp);values.append(float(critic.flatten()[0]));prev=camera
                    if len(actions)%120==0:print(args.rank,len(actions),'policy observations',flush=True)
                proc.stdin.write(f'{key}\n'.encode());proc.stdin.flush()
                ack=proc.stdout.readline().split()
                if len(ack)!=3 or ack[0]!=b'KEY' or int(ack[1])!=key:
                    raise ValueError(f'Physical keyboard differed from the policy {ack}')
                row.update(mask=key,x_applied_ns=int(ack[2]));logs.append(row)
        finally:
            try:
                proc.stdin.write(b'128\n');proc.stdin.flush();proc.stdin.close()
            except (OSError,ValueError):
                pass
            movie.release()
            if proc.wait(timeout=25)!=0:
                print('Worker exited',proc.returncode,flush=True)
    np.savez_compressed(root/'native-observers.npz',hud=np.asarray(hud_evidence),prompt=np.asarray(prompt_evidence))
    with (root/'timing.jsonl').open('w',encoding='utf8') as f:
        for row in logs:f.write(json.dumps(row)+'\n')
    np.savez_compressed(root/'behavior.npz',frames=np.asarray(frames,dtype=np.uint8),action=np.asarray(actions,dtype=np.int8),
                        logp=np.asarray(prob,dtype=np.float32),value=np.asarray(values,dtype=np.float32))
    freeze(root/'receipt.json',dict(outcome=out,policy_sha256=plan['policy'],requested_rank=args.rank,seed=args.seed,
           rank_evidence_sha256=digest(root/'chosen-rank.png'),terminal_sha256=digest(root/'terminal.png') if out=='original_continue_prompt' else None,
           observed_duration_s=(logs[-1]['monotonic_ns']-logs[0]['monotonic_ns'])/1e9 if len(logs)>1 else 0,
           physical_decisions=len(actions),direction_histogram=np.bincount(np.array(actions,dtype=np.int64).reshape(-1,4)[:,0],minlength=9).tolist(),
           role=args.role,native_observers_sha256=digest(root/'native-observers.npz'),behavior_sha256=digest(root/'behavior.npz'),timing_sha256=digest(root/'timing.jsonl'),movie_sha256=digest(root/'original-observations.mp4'),
           movie_timing='One captured frame per decision or transition, displayed at 8 fps. Source nanosecond timestamps in timing.jsonl.'))
    print(args.rank,out,len(actions),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('assets','policy','out'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--verified-menu',action='store_true')
    p.add_argument('--role',choices=('engineering-zero-shot-original','train','held'),default='engineering-zero-shot-original')
    p.add_argument('--rank',choices=('Hard','Lunatic'),required=True)
    p.add_argument('--seed',type=int,required=True)
    run(p.parse_args())
