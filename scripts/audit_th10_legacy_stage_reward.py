"""Audit native measurement coverage and awarded stage rewards on the registered TH10 training credits."""
import argparse
import json
from pathlib import Path
import numpy as np

from scripts.collect_th10_full_action import save


def audit(root,out):
    original_hash='5430fac601d645798ccf794362ec7d4cd753814f09d160247dc628b7c6943fd8'
    lineages={};seen=set()
    for manifest in sorted(root.rglob('ppo-receipts.json')):
        entries=json.loads(manifest.read_bytes())
        if not isinstance(entries,list):continue
        group=[]
        for record in entries:
            if record.get('title')!=10 or record.get('role')!='train':continue
            name=record.get('behavior') or record.get('file') or record.get('path')
            if not name:raise ValueError(f'Unspecified native trace in {manifest}')
            path=Path(name)
            if not path.is_file():path=manifest.parent/path
            report=json.loads((path.parent/'report.json').read_bytes())
            assert report['game_sha256']==original_hash
            identity=(report['policy_sha256'],report['behavior_sha256'])
            if identity in seen:continue
            seen.add(identity)
            with np.load(path) as data:
                component=data['reward_stage'];n=len(component)
                assert n==report['steps']
                visible=path.parent/'original-pixels.npz'
                group.append(dict(seed=record['seed'],policy_sha256=report['policy_sha256'],native_behavior_sha256=report['behavior_sha256'],
                    steps=n,positive_stage_steps=int((component>0).sum()),recorded_stage_reward=float(component.sum()),
                    original_policy_camera_available=visible.is_file(),physical_host=report['host'],
                    difficulty=report['source'].get('difficulty',1)))
        if group:lineages[manifest.relative_to(root).as_posix()]=group
    rows=[r for group in lineages.values() for r in group]
    summary={k:dict(credits=len(v),steps=sum(r['steps'] for r in v),stage_reward_sum=sum(r['recorded_stage_reward'] for r in v),
                    positive_stage_steps=sum(r['positive_stage_steps'] for r in v)) for k,v in lineages.items()}
    save(out,dict(scope='Unique TH10 training trajectories in native ppo-receipts.json manifests. Held and unregistered credits excluded.',
        game_sha256=original_hash,legacy_stage_register='0x472b44',
        correction='The historical native CSV, action-state arrays and reward_stage component did not retain the true progression register. A true Stage 1→2 would have deserved +2 on its original action interval; the number of missed rewards requires an independent archived-camera witness. No change to original stored returns.',
        unique_train_credits=len(rows),original_steps=sum(r['steps'] for r in rows),
        stage_reward_sum=sum(r['recorded_stage_reward'] for r in rows),
        positive_stage_steps=sum(r['positive_stage_steps'] for r in rows),
        original_cameras_available=sum(r['original_policy_camera_available'] for r in rows),summary=summary,lineages=lineages))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();audit(args.root.resolve(),args.out.resolve())
