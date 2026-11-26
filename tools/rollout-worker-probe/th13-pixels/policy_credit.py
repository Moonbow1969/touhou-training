"""Collect a matched two-credit online policy-gradient block on private X displays."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time
import numpy as np
from PIL import Image
from collect import read_hud


def features(obs):
    a=obs.astype(np.float64)/255
    prior=np.concatenate([a[:1],a[:-1]],axis=0)
    def grid(c):return c.reshape(-1,7,16,6,16).mean((2,4)).reshape(-1,42)
    return np.concatenate([np.ones((len(a),1)),grid(a),grid(np.abs(a-prior))],axis=1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ssh',required=True);p.add_argument('--known-hosts',required=True)
    p.add_argument('--prefix',required=True);p.add_argument('--remote-root',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--round',type=int,required=True)
    p.add_argument('--model-a',type=Path,required=True);p.add_argument('--model-b',type=Path,required=True)
    p.add_argument('--lanes',default='ab');p.add_argument('--seed-a',type=int);p.add_argument('--seed-b',type=int);args=p.parse_args()
    args.output.mkdir(exist_ok=True,parents=True)
    opts=['-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+args.known_hosts]
    def remote(command,**kwargs):
        return subprocess.run(['ssh',*opts,args.ssh,command],capture_output=True,check=True,**kwargs).stdout
    def run(w):
        model=args.model_a if w=='a' else args.model_b
        W=np.load(model)['weights'];assert W.shape==(3,85)
        raw=W.astype('<f8').tobytes();fingerprint=hashlib.sha256(raw).hexdigest()
        tag=f'pg-{args.round:02d}-{w}';base=f'{args.remote_root}/{w}/{tag}';local=args.output/tag
        local.mkdir(exist_ok=True)
        def run_game(cmd):return remote(shlex.join(['docker','exec','-e','DISPLAY=:66',args.prefix+w,'sh','-ec',cmd]))
        def fetch(name):return remote('cat '+shlex.quote(base+'/'+name))
        run_game(f'mkdir -p /capture/{tag}; chmod 777 /capture/{tag}')
        remote('cat > '+shlex.quote(base+'/weights.bin'),input=raw)
        run_game(f'cd /capture/{tag}; xdotool keydown r; sleep .2; xdotool keyup r; sleep 3.9; scrot --overwrite start.png')
        start=Image.open(io.BytesIO(fetch('start.png'))).convert('RGB');initial=read_hud(start)
        if not(initial['spare_lives']==2 and initial['bombs']==2 and initial['score_zero']):
            run_game(f'sleep .55; scrot --overwrite /capture/{tag}/start.png')
            start=Image.open(io.BytesIO(fetch('start.png'))).convert('RGB');initial=read_hud(start)
        if not(initial['spare_lives']==2 and initial['bombs']==2 and initial['score_zero']):raise ValueError((w,'fresh gate failed',initial))
        seed=args.seed_a if w=='a' else args.seed_b
        if seed is None:seed=253000+args.round*2+(0 if w=='a' else 1)
        start.crop((3,22,643,502)).save(local/'start.webp',lossless=True)
        plan=dict(lane=w,round=args.round,seed=seed,weights_sha256=fingerprint,model=model.name,
                  start=initial,state='collecting',reward='ending spare count minus starting spare count; observed Game Over value -3; pending death trance needs settlement; unresolved is excluded')
        (local/'credit.json').write_text(json.dumps(plan,indent=2))
        run_game(f"""cd /capture/{tag}
( for i in 1 2 3 4 5; do sleep 9.9; scrot --overwrite hud-$i.png; done ) &
snap=$!
/tmp/pixel-policy weights.bin {seed} 60 340 >actor.log 2>&1
scrot --overwrite endpoint.png
sleep 3; scrot --overwrite settle-3.png
sleep 4; scrot --overwrite settle-7.png
sleep 5; scrot --overwrite settle-12.png
xdotool keydown Escape; sleep .18; xdotool keyup Escape; sleep .65
wait "$snap"
tar -czf /capture/{tag}.tgz pixels.u8 wall.ns behavior.tsv endpoint.png settle-3.png settle-7.png settle-12.png hud-*.png
""")
        archive=remote('cat '+shlex.quote(base+'.tgz'))
        with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as tar:
            def buf(n):return tar.extractfile(n).read()
            obs=np.frombuffer(buf('pixels.u8'),np.uint8).reshape(-1,112,96)
            clocks=np.fromstring(buf('wall.ns').decode(),np.int64,sep=' ')
            beh=np.genfromtxt(io.BytesIO(buf('behavior.tsv')),names=True,delimiter='\t')
            a=beh['action'].astype(int);probs=np.column_stack([beh[k] for k in ['p_left','p_neutral','p_right']])
            F=features(obs[:-1]);logits=F@W.T
            check=np.exp(logits-logits.max(1)[:,None]);check/=check.sum(1)[:,None]
            assert np.allclose(probs,check,atol=2e-9,rtol=2e-9)
            assert np.allclose(np.log(probs[np.arange(len(a)),a]),beh['logp'])
            onsets=beh['onset_ns'].astype('int64')
            assert len(obs)==len(a)+1 and np.all((clocks[:-1]<onsets)&(onsets<clocks[1:]))
            np.savez_compressed(local/'behavior.npz',obs=obs,action=a.astype('int8'),prob=probs,logp=beh['logp'],monotonic_ns=clocks,action_onset_ns=onsets)
            hud={}
            for n in [f'hud-{i}' for i in range(1,6)]+['endpoint','settle-3','settle-7','settle-12']:
                im=Image.open(io.BytesIO(buf(n+'.png'))).convert('RGB')
                hud[n]=read_hud(im);im.crop((3,22,643,502)).save(local/(n+'.webp'),lossless=True,method=2)
        plan.update(state='awaiting_visual_endpoint_classification',decisions=len(a),actual_seconds=round((clocks[-1]-clocks[0])*1e-9,4),hud=hud,action_count=np.bincount(a,minlength=3).tolist())
        (local/'credit.json').write_text(json.dumps(plan,indent=2)+'\n')
        print(tag,plan['decisions'],[(k,hud[k]['spare_lives']) for k in ['endpoint','settle-3','settle-7','settle-12']],flush=True)
    with ThreadPoolExecutor(len(args.lanes)) as ex:list(ex.map(run,args.lanes))

if __name__=='__main__':main()
