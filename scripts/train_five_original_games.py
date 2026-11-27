"""Fit one ten-stratum PPO actor using exact same-behavior native trajectories from five games."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.checkpoint_storage import digest


def freeze(path, value):
    if path.exists() and json.loads(path.read_bytes()) != value:
        raise ValueError(f'Changed frozen experiment {path}')
    if not path.exists():
        path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True,type=Path)
    args=parser.parse_args()
    data=args.data.resolve();root=data/'five-game-online';root.mkdir(exist_ok=True)
    parent=data/'specialist-vs-shared/merged-equal'
    actor=parent/'policy.ts';fingerprint=digest(actor)
    four=data/'four-game-merged-online';one=data/'th11-original-ppo'
    originals=[four/'corrected/credits.json',one/'training-credits.json']
    plans=[json.loads((d/'camera-protocol.json').read_bytes()) for d in (four,one)]
    common=plans[0]
    if any(p['behavior_sha256']!=fingerprint for p in plans):
        raise ValueError('The two original acquisitions used different deployed parent policies')
    train={k:v for p in plans for k,v in p['train_seeds'].items()}
    held={k:ss[:16] for p in plans for k,ss in p['eval_seeds'].items()}
    held['12']=list(range(1441251,1441267))
    ranks={k:{str(s):2 if s%2 else 3 for s in train[k]+held[k]} for k in train}
    plan=dict(name='Joint PPO on five original Touhou games', reference_arm='merged-sft', candidate='five-ppo',
              train_seeds=train,eval_seeds=held,rank_by_seed=ranks,behavior_sha256=fingerprint,
              games={k:v for p in plans for k,v in p['games'].items()},balanced_title_rank=True,
              learning_rates=common['learning_rates'],ppo=common['ppo'],controls=common['controls'],
              starting_actor='Uniform average of four same-parent human SFT specialists. Zero initial value head.',
              source_manifests={f.relative_to(data).as_posix():digest(f) for f in originals},
              design='Use all 48 distinct physical original credits from the same frozen actor and the original native rewards. Compare with the four-game and TH11-only sibling fits to the corresponding subsets. Each of the ten game by difficulty strata receives equal optimizer weight. These credits already appear in the physical collection ledger.',
              evaluation='Eight pairs per title and difficulty. TH10, TH13, TH17 use the first sixteen registered four-game evaluation seeds and their frozen parent credits. TH11 uses all its specialist evaluation seeds and parent credits. TH12 uses sixteen new paired seeds on the repaired native-four-tick controller. Each pair is matched within one original build, physical host and lane. The four-game and TH11 seeds are shared comparisons with previously run checkpoints; TH12 has an independent cohort.')
    freeze(root/'camera-protocol.json',plan)
    covered=json.loads((four/'th12-train-qualified-coverage.json').read_bytes())['credits']
    approved={v['receipt_sha256'] for v in covered if v['coverage']=='qualified' and v['evidence_sha256']}
    directory=root/'training';directory.mkdir(exist_ok=True)
    rows=[];seen=set()
    for file in originals:
        for r in json.loads(file.read_bytes()):
            title,seed=r['title'],r['seed']
            if (title,seed) in seen or r['policy_sha256']!=fingerprint or r['role']!='train':
                raise ValueError('Duplicate credit, wrong policy or non-training episode')
            if r['difficulty']!=ranks[str(title)][str(seed)] or not r['difficulty_evidence_sha256']:
                raise ValueError('Native difficulty disagrees with registration')
            if title==12 and r['source_log_sha256'] not in approved:raise ValueError('Unqualified TH12 control')
            source=file.parent/r['file'];target=directory/f'th{title}-{seed}.npz'
            if digest(source)!=r['sha256']:raise ValueError('Changed original NPZ')
            if not target.exists():os.link(source,target)
            if digest(target)!=r['sha256']:raise ValueError('Changed linked NPZ')
            seen.add((title,seed))
            rows.append(dict(r,file=target.name,original_manifest=file.relative_to(data).as_posix()))
    if len(seen)!=48 or len(approved)!=8:raise ValueError('Missing fresh credits or qualification')
    manifest=directory/'credits.json';freeze(manifest,rows)
    if not (root/'policy/report.json').is_file():
        subprocess.run([sys.executable,'-u','-m','scripts.train_full_resolution_camera_ppo','--manifest',manifest,
                        '--protocol',root/'camera-protocol.json','--behavior',actor,'--sft',parent/'camera-policy.pt',
                        '--out',root/'policy'],check=True)
    report=json.loads((root/'policy/report.json').read_bytes())
    if (report['native_behavior_sha256']!=fingerprint or report['policy_sha256']!=digest(root/'policy/policy.ts')
            or len(report['equal_weight_strata'])!=10 or len(report['credits'])!=48):
        raise ValueError('Unexpected five-title learner results')
    print('Five original titles',report['steps'],'on-policy decisions',report['policy_sha256'],flush=True)


if __name__=='__main__':
    main()
