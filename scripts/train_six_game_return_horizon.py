"""Collect one new six-title on-policy population and compare two actor credit-assignment horizons."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import os
import subprocess
import sys
import time

import numpy as np
import torch
from scripts.check_five_title_critic_returns import discounted_returns
from scripts.checkpoint_storage import digest
from scripts.continue_th08_online_extension import atomic_json, launch_after_entry_failures, selected_credit
from scripts.run_registered_camera_update import wait_th12_coverage
from scripts.score_original_camera_run import records, verified_th10_entry
from scripts.train_crossfit_five_game_ppo import build_baselines
from scripts.train_five_original_games import freeze
from scripts.train_shared_full_ppo import calculate_gae
from scripts.train_six_original_games import hardlink, read, wait
from scripts.th10_playable_actions import crop_packed, verify_source as th10_end

TITLES=(8,10,11,12,13,17)

def register(d):
    r=d/'six-game-return-horizon';r.mkdir(exist_ok=True)
    parent=d/'six-game-online/joint-six/policy'
    previous=read(d/'six-game-online/protocol.json')
    five=read(d/'five-game-generation3/camera-protocol.json')
    games=dict(five['games'],TH08='Original trial v0.03, Border Team')
    train={str(t):[1710000+t*100+i for i in range(1,33)] for t in TITLES}
    held={str(t):[1710000+t*100+i for i in range(51,67)] for t in TITLES}
    ranks={t:{str(s):2 if s%2 else 3 for s in train[t]+held[t]} for t in train}
    plan=dict(name='Six-title actor credit assignment from fresh full-stock games',games=games,
        behavior_sha256=digest(parent/'policy.ts'),parent_actor_critic_sha256=digest(parent/'camera-ppo.pt'),
        training_arm='six-ppo',reference_arm='six-gae985',candidate='six-return',balanced_title_rank=True,
        train_seeds=train,eval_seeds=held,rank_by_seed=ranks,ppo=five['ppo'],controls=five['controls'],
        learning_rates=five['learning_rates'],th08_reward=previous['reward'],
        rationale='The native critic overestimated observed terminal returns on ten previous game/rank cells. With gamma .999 and actor lambda .985 the trace weight falls by half after about 43 decisions. Test whether complete terminal reward-to-go gives better ordinary full-stock outcomes from the same real experiences.',
        training='Collect sixteen fresh behavior-policy credits per game and rank. Both learners use the exact same six-title initial weights, 192 physical games, four-fold cross-fitted elapsed-decision baseline, rewards, native GAE critic targets, four epochs, optimizer seeds and equal strata weights. Compare actor lambda .985 with 1. For lambda 1 require numerical equivalence to discounted Monte Carlo return minus the elapsed baseline. Game identity, elapsed decisions, ranks, rewards and stock data are excluded from the deployed actor.',
        technical='Keep each started technical failure, evidence and reason. Rerun exactly the same assigned controller seed, rank, host and arm for a truncated hardware/entry failure. Do not replace based on scored outcomes. Confirm the first original continue prompt before TH08 training. For TH12 require native consumption and continuous control. A missing technical allocation keeps the cohort incomplete.',
        evaluation='Eight new matched seeds, two separate ordinary-stock original games per seed and difficulty in all six titles. Alternate arm order within rank. Same host and build within pair, uncontrolled in-game RNG. Primary Stage 2 rate. Secondary remaining lives and Bombs or Power, plus per-title native progress. Original commercial and trial builds are distinguished. Queue collection and testing behind previously registered uses on each physical lane.')
    freeze(r/'camera-protocol.json',plan)
    five_view=dict(plan,train_seeds={t:ss for t,ss in train.items() if t!='8'},eval_seeds={t:ss for t,ss in held.items() if t!='8'})
    freeze(r/'five-title-packing.json',five_view)
    slots=plan['train_seeds']['8']
    hosts={str(s):('Gamer' if (j//2)%2==0 else 'Entropy') for j,s in enumerate(slots)}
    freeze(r/'th08-host-allocation.json',dict(registration=digest(r/'camera-protocol.json'),by_seed=hosts,
        host_clock_check=digest(d/'th08-multiteam-online/default-team-host-clock-comparison.json'),
        allocation='Before any TH08 training input for this study, allocate eight Hard and eight Lunatic games to each of two qualified original-game hosts. Both use Border Team, the same G1 six-title behavior and the same original executable. Hardened title entry on Entropy and visible-window focus on Gamer apply before policy control. Preserve technical attempts and retry the same physical seed on the same host. The GAE and Monte Carlo learners receive identical source bytes, including the host split.'))
    print('Registered',r.name,digest(r/'camera-protocol.json'),flush=True)
    return r,plan,parent


def th08_lane(d,r,plan,parent,assets,rank):
    # Reserve the existing Gamer rank lane until its paired retention allocation has finished.
    for s in read(d/'six-game-retention/camera-protocol.json')['eval_seeds']['8']:
        if (2 if s%2 else 3)==rank:wait(d/'six-game-retention/th08/evaluation'/f'pair-{s}.json')
    model=torch.jit.load(str(parent/'policy.ts'),map_location='cpu').eval()
    native=r/'native08';native.mkdir(exist_ok=True)
    selected=r/'selected08';selected.mkdir(exist_ok=True)
    label={2:'Hard',3:'Lunatic'}[rank]
    allocation=read(r/'th08-host-allocation.json')
    for s in plan['train_seeds']['8']:
        if allocation['by_seed'][str(s)]!='Gamer' or plan['rank_by_seed']['8'][str(s)]!=rank or (selected/f'{s}.json').exists():continue
        base=native/f'train-six-ppo-{s}'
        attempts=[]
        for i in range(3):
            folder=base if i==0 else base.with_name(base.name+f'-technical-{i+1}')
            folder,entries=launch_after_entry_failures(assets,parent/'policy.ts',folder,label,s,'train')
            item,bad=selected_credit(r,folder,model,plan['behavior_sha256'],s,label,'train',r/'th08-training')
            if bad:
                attempts.append(dict(source=folder.name,reason=bad,zero_action_entries=entries))
                atomic_json(selected/f'{s}-attempt-{i+1}.json',attempts[-1]);continue
            item.update(slot=s,physical_host='Gamer',technical_games=attempts,entry_failures=entries)
            atomic_json(selected/f'{s}.json',item)
            break
        else:raise RuntimeError(f'Unfilled TH08 slot {s}')
        print('Qualified fresh six-title policy interaction',s,item['audit']['decisions'],flush=True)


def fit(d,r,plan,parent):
    def run(module,*args):
        subprocess.run([sys.executable,'-u','-m','scripts.'+module,*map(str,args)],check=True)
    source=r/'five-title-packing.json';five=read(source)
    wanted={(int(t),s) for t,seeds in five['train_seeds'].items() for s in seeds}
    last=-1
    while True:
        seen={}
        for t,path in records(r,True,True):
            try:receipt=read(path)
            except (json.JSONDecodeError,PermissionError):continue
            key=t,receipt.get('seed')
            if key not in wanted:continue
            if key in seen or receipt['policy_sha256']!=plan['behavior_sha256'] or not receipt.get('terminal',True) or receipt.get('continues',0):
                raise ValueError('Unqualified physical training credit')
            if t==10 and verified_th10_entry(receipt,path.parent)[0] is None:raise ValueError('Missing native TH10 stage')
            seen[key]=path
        if len(seen)!=last:print('Fresh five-title training credits',len(seen),'/ 160',flush=True);last=len(seen)
        if set(seen)==wanted:break
        time.sleep(30)
    wait_th12_coverage(r/'th12-train-qualified-coverage.json',[f for (t,_),f in seen.items() if t==12])
    if not (r/'camera-batch/credits.json').exists():run('pack_camera_ppo_rollouts','--root',r,'--round2','--protocol',source,'--behavior',parent/'policy.ts','--out',r/'camera-batch')
    if not (r/'corrected/clock-audit.json').exists():run('normalize_th13_survival_clock','--original',r,'--out',r/'corrected')
    parts=[]
    boundaries=[]
    for row in read(r/'corrected/credits.json'):
        path=r/'corrected'/row['file']
        if row['title']==10:
            audit=th10_end(seen[(10,row['seed'])])
            if audit['report_sha256']!=row['source_log_sha256'] or audit['trace_sha256']!=row['source_actions_sha256'] or digest(path)!=row['sha256']:
                raise ValueError('TH10 terminal audit and collected rollout differ')
            if audit['dropped_steps']:
                target=r/'th10-playable'/row['file'];target.parent.mkdir(exist_ok=True)
                with np.load(path,allow_pickle=False) as z:
                    arrays=crop_packed(dict(z),audit)
                if not target.exists():np.savez_compressed(target,**arrays)
                with np.load(target,allow_pickle=False) as z:
                    if set(z.files)!=set(arrays) or any(not np.array_equal(z[k],v) for k,v in arrays.items()):
                        raise ValueError('Retained TH10 training prefix differs from the original')
                row=dict(row,sha256=digest(target),before_terminal_audit_sha256=row['sha256'])
                path=target
            boundaries.append(dict(seed=row['seed'],optimizer_input=row['sha256'],evidence=audit))
        parts.append((row,path))
    if len(boundaries)!=32:raise ValueError('TH10 training boundary check missed a credit')
    freeze(r/'th10-optimizer-terminal-boundaries.json',boundaries)
    assignment=read(r/'th08-host-allocation.json')['by_seed']
    for s in plan['train_seeds']['8']:
        f=r/'selected08'/f'{s}.json';wait(f);selected=read(f);row=selected['record']
        if selected['physical_host']!=assignment[str(s)] or selected['slot']!=s or row['seed']!=s or row['difficulty']!=plan['rank_by_seed']['8'][str(s)]:
            raise ValueError('Unverified TH08 host, seed or rank allocation')
        parts.append((row,r/'th08-training'/row['file']))
    bundle=r/'optimizer-input';bundle.mkdir(exist_ok=True)
    rows=[]
    for row,path in parts:
        name=f"th{row['title']}-{row['seed']}.npz"
        hardlink(path,bundle/name,row['sha256']);rows.append(dict(row,file=name))
    manifest=bundle/'credits.json';freeze(manifest,rows)
    baselines=build_baselines(rows,manifest,r,titles=TITLES)
    # Reconstruct a sparse terminal reward with unrelated baseline values to test the recurrence.
    rng=np.random.default_rng(1710000);v=rng.normal(size=1500).astype(np.float32)
    reward=np.zeros_like(v);reward[[200,617,1499]]=[-1,.03,-1]
    eq=np.max(abs(calculate_gae(reward,v,.999,1.)-(discounted_returns(reward,.999)-v)))
    if eq>2e-5:raise ValueError('Actor lambda=1 disagrees with full Monte Carlo')
    freeze(r/'estimator-check.json',dict(max_error=float(eq),train_manifest=digest(manifest),baselines=digest(baselines)))
    for arm,lam in ((plan['reference_arm'],.985),(plan['candidate'],1.)):
        out=r/arm;out.mkdir(exist_ok=True);target=out/'fit-protocol.json'
        freeze(target,dict(plan,registered=digest(r/'camera-protocol.json'),actor_gae_lambda=lam,actor_advantage_baselines_sha256=digest(baselines)))
        if not (out/'policy/report.json').exists():
            run('train_full_resolution_camera_ppo','--protocol',target,'--manifest',manifest,'--behavior',parent/'policy.ts',
                '--parent-actor-critic',parent/'camera-ppo.pt','--actor-advantage-baselines',baselines,'--out',out/'policy')
        result=read(out/'policy/report.json')
        if result['manifest_sha256']!=digest(manifest) or result['policy_sha256']!=digest(out/'policy/policy.ts') or len(result['credits'])!=192:
            raise ValueError('Unverified six-game horizon fit')
        print('Fitted',arm,result['steps'],result['policy_sha256'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--assets',type=Path)
    p.add_argument('--role',choices=('register','th08','fit'),required=True)
    a=p.parse_args();d=a.data.resolve();r,plan,parent=register(d)
    if a.role=='fit':fit(d,r,plan,parent)
    if a.role=='th08':
        torch.set_num_threads(1)
        if not a.assets:p.error('TH08 collection needs the local original game assets')
        with ThreadPoolExecutor(2) as workers:
            tasks=[workers.submit(th08_lane,d,r,plan,parent,a.assets.resolve(),rank) for rank in (2,3)]
            for t in tasks:t.result()


if __name__=='__main__':
    main()
