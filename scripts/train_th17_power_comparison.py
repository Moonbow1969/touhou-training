"""Train a same-data four-title PPO sibling with native TH17 high-water Power rewards."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from scripts.checkpoint_storage import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    d=parser.parse_args().data.resolve()
    base=d/'four-game-ppo-generation2'
    root=d/'th17-power-highwater'
    out=root/'corrected';out.mkdir(exist_ok=True)
    parent=d/'four-game-online/policy'
    plan=json.loads((root/'camera-protocol.json').read_bytes())
    dataset=base/'corrected/credits.json'
    if digest(dataset)!=plan['on_policy_source_manifest_sha256'] or digest(parent/'policy.ts')!=plan['behavior_sha256']:
        raise ValueError('Source G2 behavior or corrected reward batch differs')
    overlay_dir=root/'native17/overlays'
    manifest=overlay_dir/'manifest.json'
    overlays=json.loads(manifest.read_bytes())['credits']
    index={r['seed']:r for r in overlays}
    if len(index)!=len(overlays) or set(index)!=set(plan['train_seeds']['17']):raise ValueError('TH17 overlay seeds differ')
    rows=json.loads(dataset.read_bytes())
    audit=[]
    for row in rows:
        source=dataset.parent/row['file'];target=out/row['file']
        if digest(source)!=row['sha256']:raise ValueError('Source PPO trajectory changed')
        if row['title']!=17:
            if not target.exists():
                try:os.link(source,target)
                except OSError:shutil.copy2(source,target)
            if digest(target)!=row['sha256']:raise ValueError('Non-TH17 trajectory differs')
            continue
        o=index[row['seed']];f=overlay_dir/o['file']
        if o['rank']!=row['difficulty'] or digest(f)!=o['overlay_sha256'] or row['source_actions_sha256']!=o['behavior_sha256']:
            raise ValueError('Native Power overlay does not match the source credit')
        with np.load(source,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        with np.load(f,allow_pickle=False) as z:
            bonus=z['reward_bonus']; original=z['reward_original']
        if (original.shape!=bonus.shape or bonus.dtype!=np.float32 or len(bonus)!=o['actions']
            or not np.array_equal(original,arrays['reward']) or not np.all(np.isfinite(bonus))
            or bonus.min()<0 or bonus.sum()>.150001 or abs(bonus.sum()-o['reward_bonus_sum'])>1e-6):
            raise ValueError('Invalid or misaligned pickup reward')
        arrays['reward_before_power_bonus']=original
        arrays['power_bonus']=bonus
        arrays['reward']=original+bonus
        if not target.exists():np.savez_compressed(target,**arrays)
        else:
            with np.load(target,allow_pickle=False) as existing:
                if set(existing.files)!=set(arrays) or any(not np.array_equal(existing[k],v) for k,v in arrays.items()):
                    raise ValueError('Previous packed overlay differs')
        row['sha256']=digest(target)
        audit.append(dict(seed=row['seed'],difficulty=row['difficulty'],bonus=float(bonus.sum()),
                          genuine_new_high_decisions=int(np.count_nonzero(bonus)),overlay_sha256=digest(f)))
    (out/'credits.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'reward-audit.json').write_text(json.dumps(dict(overlays_sha256=digest(manifest),rows=audit),indent=2)+'\n')
    if not (root/'policy/report.json').exists():
        subprocess.run(list(map(str,[sys.executable,'-u','-m','scripts.train_full_resolution_camera_ppo',
            '--manifest',out/'credits.json','--protocol',root/'camera-protocol.json','--behavior',parent/'policy.ts',
            '--parent-actor-critic',parent/'camera-ppo.pt','--out',root/'policy'])),check=True)
    report=json.loads((root/'policy/report.json').read_bytes())
    original=json.loads((base/'policy/report.json').read_bytes())
    if (report['steps']!=original['steps'] or report['parent_actor_critic_sha256']!=original['parent_actor_critic_sha256']
        or report['policy_sha256']!=digest(root/'policy/policy.ts')):raise ValueError('Unmatched learner comparison')
    print(report['policy_sha256'],len(audit),report['steps'],flush=True)


if __name__=='__main__':
    main()
