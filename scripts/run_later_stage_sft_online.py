"""Deploy the later-stage human SFT to two original titles, fit on-policy PPO and run full held pairs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
from scripts.checkpoint_storage import digest
from scripts.qualified_th11_stage2 import read_credit
from scripts.run_th10_camera_cohort import collect
from scripts.run_th11_trial_credit import Lane,SRC,finish,run as original11
from scripts.train_five_original_games import freeze

def read(f):return json.loads(f.read_bytes())
def wait(f):
    while not f.is_file():time.sleep(10)
def call(module,*args):
    subprocess.run([sys.executable,'-u','-m','scripts.'+module,*map(str,args)],check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--workers',type=Path,required=True)
    parser.add_argument('--recover-lane',type=int,choices=(10,11))
    a=parser.parse_args();d=a.data.resolve();workers=a.workers.resolve()
    study=d/'later-human-online';study.mkdir(exist_ok=True)
    source=d/'four-title-later-stage-sft/policy';root=d/'six-game-return-horizon'
    parent=read(source/'report.json');actor=digest(source/'policy.ts')
    if actor!=parent['torchscript_sha256'] or sum(r['decisions'] for r in parent['train'])!=231511:raise ValueError('Changed human SFT')
    prior=read(d/'five-game-generation3/camera-protocol.json')
    tlist=(10,11);train={str(t):list(range(1760001+t*100,1760009+t*100)) for t in tlist}
    held={str(t):list(range(1760051+t*100,1760067+t*100)) for t in tlist}
    ranks={t:{str(s):2 if s%2 else 3 for s in train[t]+held[t]} for t in train}
    ref,alt='later-sft','later-ppo';protocol=study/'camera-protocol.json'
    plan=dict(name='Original full-action PPO from the 68-stage human visual SFT',reference_arm=ref,candidate=alt,
        games={f'TH{t}':prior['games'][f'TH{t}'] for t in tlist},train_seeds=train,eval_seeds=held,rank_by_seed=ranks,
        behavior_sha256=actor,native_executable_sha256=prior['native_executable_sha256'],balanced_title_rank=True,
        verify_th10_playable_end=True,learning_rates=prior['learning_rates'],ppo=prior['ppo'],controls=prior['controls'],
        training='Eight fresh ordinary complete original credits per title, four per difficulty. One shared visual actor from human SFT with zero initial critic. TH10 supplied human cameras to this SFT. TH11 had no direct human camera demonstrations in this SFT fit.',
        evaluation='Use eight matched pairs per title and difficulty. Compare direct human SFT with its own two-title PPO child. All movement, focus, shot and Bomb choices are learned. Each paired credit shares host, build, controller seed and difficulty, with ordinary stock and no Continue. Native RNG is uncontrolled. Primary Stage 2, secondary spare lives, Power and native progress or score. Alternate arm order within each rank. Keep technical attempts.')
    freeze(protocol,plan)
    qualification=study/'stage2-qualification.json';src=d/'th11-original-ppo/stage2-qualification.json'
    if not qualification.exists():shutil.copy2(src,qualification)
    if digest(qualification)!=digest(src):raise ValueError('TH11 original qualification differs')
    models={ref:source/'policy.ts',alt:study/'policy/policy.ts'}
    def physical(t,seed,arm,role):
        rank=ranks[str(t)][str(seed)];model=models[arm]
        if role=='train' and digest(model)!=actor:raise ValueError('Off-policy training input')
        if t==10:
            options=SimpleNamespace(workers=workers,host='momentum',difficulty=rank,stage_sensor='qualified-v2-clock')
            out=study/'native10'/role/arm/str(seed)
            found=list(out.rglob('report.json')) if out.exists() else []
            if not found:
                collect(options,role+'-'+arm,model,[seed],out,1,True)
                found=list(out.rglob('report.json'))
            if len(found)!=1:raise ValueError('Expected exactly one physical TH10 receipt for this arm')
            record=read(found[0])
        else:
            out=study/'native11'/f'{role}-{arm}-{seed}'
            if not (out/'receipt.json').is_file():
                active=Lane(d,workers,out.name,out)
                try:
                    if not (out/'build.json').exists():
                        original11(d,workers,seed,rank,role=role,arm=arm,model=model,out=out,expected_sha=digest(model))
                except (OSError,subprocess.SubprocessError) as e:
                    print('Recover original TH11 connection',seed,type(e).__name__,flush=True)
                if not (out/'receipt.json').is_file():
                    camera=SRC.parents[1]/'full_action'
                    files=[SRC/'camera-credit.c',SRC/'native-stock.c',SRC/'launch-game.sh',camera/'camera_actor.h',camera/'camera_service.py',SRC.parent/'th13-pixels/x11_pixels.h']
                    for i in range(120):
                        try:done=active.remote('test ! -s '+shlex.quote(active.dir+'/done')+' || echo done').strip()==b'done'
                        except (OSError,subprocess.SubprocessError):done=False
                        if done:
                            finish(active,digest(model),seed,rank,files,role,arm)
                            break
                        time.sleep(5)
                    else:raise RuntimeError(f'Original TH11 credit has no completed native endpoint after recovery: {seed}')
            record=read(out/'receipt.json');read_credit(out)
            if role=='train':
                with np.load(out/'trajectory.npz',allow_pickle=False) as z:
                    if np.any(z['value']):raise ValueError('SFT actor should have a zero critic')
        if record['seed']!=seed or record['difficulty']!=rank or record['policy_sha256']!=digest(model) or not record['terminal'] or record['continues']:
            raise ValueError('Unexpected original-game receipt')
        print(t,role,seed,arm,record['steps'],flush=True)
    def lane(t):
        for s in train[str(t)]:
            # Reserve the previous study's two-lane evaluation once its learner is ready.
            if (root/'six-return/policy/report.json').is_file():
                wait(root/(f'th{t}-held-native-records.zip' if t==11 else 'native10/held-qualified-native-ranks.json'))
            physical(t,s,ref,'train')
        wait(study/'policy/report.json')
        for i,s in enumerate(held[str(t)]):
            if (root/'six-return/policy/report.json').is_file():
                wait(root/(f'th{t}-held-native-records.zip' if t==11 else 'native10/held-qualified-native-ranks.json'))
            for arm in ((ref,alt) if (i//2)%2==0 else (alt,ref)):physical(t,s,arm,'held')
    if a.recover_lane:
        lane(a.recover_lane)
        return
    with ThreadPoolExecutor(3) as pool:
        jobs=[pool.submit(lane,t) for t in tlist]
        jobs.append(pool.submit(call,'run_registered_camera_update','--root',study,'--parent',source,'--protocol',protocol,'--from-sft'))
        def log_failure(job):
            if job.exception() is not None:print('Worker failed',repr(job.exception()),file=sys.stderr,flush=True)
        for job in jobs:job.add_done_callback(log_failure)
        for j in jobs:j.result()
    call('report_whole_game_controls','--root',study)
    for rank in ('Hard','Lunatic'):
        call('render_cross_title_credits','--root',study,'--out',study/'evaluation'/f'first-{rank.lower()}-native-credits.mp4','--difficulty',rank,'--titles',10,11)


if __name__=='__main__':
    main()
