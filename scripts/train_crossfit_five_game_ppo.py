"""Compare two PPO actor advantage estimators on exactly the same five-game training credits."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.check_five_title_critic_returns import discounted_returns, ridge
from scripts.checkpoint_storage import digest
from scripts.th12_playable_actions import playable_length
from scripts.train_five_original_games import freeze


def read(path):
    return json.loads(path.read_bytes())


def build_baselines(rows, manifest, directory, titles=(10,11,12,13,17)):
    groups = {}
    for row in rows:
        path = manifest.parent / row['file']
        if row['role'] != 'train' or digest(path) != row['sha256']:
            raise ValueError('Unqualified source for baseline')
        with np.load(path, allow_pickle=False) as data:
            rewards = data['reward'].reshape(-1)
            n = playable_length(row['title'], data.get('native_action_eligible'), rewards, row.get('postterminal_input_boundary'))
            target = discounted_returns(rewards[:n], .999)
        age = np.arange(n, dtype=np.float64)
        clock = np.column_stack([np.log1p(age)] + [np.minimum(age/t, 1.) for t in (32,64,128,256,512)])
        groups.setdefault((row['title'],row['difficulty']), []).append(dict(seed=row['seed'], y=target, clock=clock))
    baselines = {}
    audit = []
    for (title,rank), entries in sorted(groups.items()):
        ordered = sorted(entries, key=lambda row:row['seed'])
        if len(entries) != 16:
            raise ValueError('Expected sixteen original training games in each cell')
        ids = np.random.default_rng(290811+title*10+rank).permutation(len(entries))
        for indices in np.array_split(ids,4):
            train = [row for j,row in enumerate(ordered) if j not in indices]
            held = [ordered[j] for j in indices]
            for row, pred in zip(held, ridge(train,'clock',held),strict=True):
                baselines[f"th{title}-{row['seed']}"] = pred.astype(np.float32)
            audit.append(dict(title=title,rank=rank,training=[r['seed'] for r in train],predicted=[r['seed'] for r in held]))
    if set(groups)!={(t,r) for t in titles for r in (2,3)} or len(audit)!=8*len(titles) or len(baselines)!=32*len(titles):
        raise ValueError('Missing registered title, difficulty or credit')
    archive = directory/'actor-baselines.npz'
    if not archive.exists():
        np.savez_compressed(archive, **baselines)
    with np.load(archive,allow_pickle=False) as stored:
        if set(stored.files)!=set(baselines) or any(not np.array_equal(stored[k],v) for k,v in baselines.items()):
            raise ValueError('Out-of-fold actor targets changed')
    freeze(directory/'folds.json',audit)
    return archive


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    a=parser.parse_args()
    d=a.data.resolve()
    root=d/'five-game-crossfit-advantages'
    root.mkdir(exist_ok=True)
    source=d/'five-game-generation3'
    parent=d/'five-game-generation2/policy'
    manifest=source/'corrected/credits.json'
    original=source/'effective-train-protocol.json'
    plan=read(original)
    reference=source/'policy/policy.ts'
    if read(source/'policy/report.json')['manifest_sha256']!=digest(manifest):
        raise ValueError('Expected identical generation-three sources')
    eval_seeds={t:[1680000+100*int(t)+i for i in range(51,67)] for t in plan['train_seeds']}
    ranks={t:{**plan['rank_by_seed'][t],**{str(s):2 if s%2 else 3 for s in seeds}} for t,seeds in eval_seeds.items()}
    registration=dict(name='Cross-fitted training-time actor advantages on five original games',games=plan['games'],
        reference_arm='five-ppo3',candidate='crossfit-ppo',train_seeds=plan['train_seeds'],eval_seeds=eval_seeds,
        rank_by_seed=ranks,balanced_title_rank=True,controls=plan['controls'],ppo=plan['ppo'],
        learning_rates=plan['learning_rates'],behavior_sha256=digest(parent/'policy.ts'),reference_sha256=digest(reference),
        training_manifest_sha256=digest(manifest),original_train_protocol_sha256=digest(original),
        estimator='Within each title and rank split the 16 complete training trajectories into four folds using RNG 290811+10*title+rank. Predict each trajectory using only the other twelve. Ordinary ridge lambda .02 per training transition, target discounted Monte Carlo reward with gamma .999, causal elapsed decision features log1p(t) and min(t/k,1) for k=32,64,128,256,512. Use that baseline only in actor GAE at lambda .985. Retain the original G3 value targets and compare identical sampling, architecture, and native behavior evidence.',
        evaluation='Eight independent matched policy seed pairs per title and difficulty. Each pair shares host and game build. In-game RNG is uncontrolled. Primary Stage 2 reach without a continue. Secondary native progress within each title, remaining spare lives, Bombs or Power, confirmed casts and genuine endpoints. No assistance. No result-dependent replacement. This is a follow-up prompted by training diagnostics. Evaluation outcomes are fresh.')
    freeze(root/'camera-protocol.json',registration)
    archive=build_baselines(read(manifest),manifest,root)
    fit=root/'fit-protocol.json'
    freeze(fit,dict(registration,actor_advantage_baselines_sha256=digest(archive)))
    if not (root/'policy/report.json').exists():
        subprocess.run([sys.executable,'-u','-m','scripts.train_full_resolution_camera_ppo','--manifest',manifest,
            '--protocol',fit,'--behavior',parent/'policy.ts','--parent-actor-critic',parent/'camera-ppo.pt',
            '--actor-advantage-baselines',archive,'--out',root/'policy'],check=True)
    report=read(root/'policy/report.json')
    if report['steps']!=78740 or report['policy_sha256']!=digest(root/'policy/policy.ts') or report['actor_advantage_baselines_sha256']!=digest(archive):
        raise ValueError('Cross-fit learner lineage differs')
    print('Verified five-game advantage comparison',report['policy_sha256'],flush=True)


if __name__=='__main__':
    main()
