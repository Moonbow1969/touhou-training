"""Derive privileged imitation labels only from prospectively designated native development credits."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .actor import AnticipatoryContact,MovementStudent
from .evaluate import WEIGHT,PINNED
from .ppo import ROLE,RESERVED
from .prepare import SOURCE,dataset
from scripts.checkpoint_storage import digest
from scripts.relative_phase_student import RelativeActor
from scripts.train_full_policy import publish_json


def eligible(report):
    config=report['specification']
    if report.get('status')!='complete' or config.get('development',{}).get('role')!=ROLE:
        raise ValueError('Expected a completed run explicitly designated as development before collection.')
    ids=[t['policy_seed'] for t in config['schedule']]
    if RESERVED.intersection(ids) or len(set(ids))!=len(ids) or not ids:
        raise ValueError('An evaluation seed or duplicate credit was submitted as development.')
    actor=config['policies']['development_joint_ppo']
    if actor.get('role')!=ROLE or actor.get('source_weights_sha256')!=PINNED[WEIGHT]:
        raise ValueError('The source learner is not the prospective warm-start policy.')
    if [episode['trial'] for episode in report['episodes']]!=config['schedule']:
        raise ValueError('The development credit list differs from the declared schedule.')
    return ids


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development-report',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True,help='Fresh private derived dataset directory.')
    a=p.parse_args();torch.set_num_threads(1)
    report=json.loads(a.development_report.read_text(encoding='utf-8'));seeds=eligible(report)
    weights=torch.load(WEIGHT,weights_only=True,map_location='cpu')
    if digest(WEIGHT)!=PINNED[WEIGHT]:
        raise ValueError('The frozen initialization changed.')
    relative=RelativeActor().eval();relative.load_state_dict(weights['state_dict'])
    contact=AnticipatoryContact().eval();contact.load_state_dict(weights['contact'])
    move=MovementStudent().eval();move.load_state_dict(weights['movement'])
    a.out.mkdir(parents=True,exist_ok=False)
    summary=dict(status='building',role='development_only_privileged_relabels',scheduled_seeds=seeds,
                 source_report_sha256=digest(a.development_report),initializer_sha256=digest(WEIGHT),credits=[])
    publish_json(a.out/'manifest.json',summary)
    for ep in report['episodes']:
        d=ep['result']['transition_shard'];trail=a.development_report.parent/d['path']
        if digest(trail)!=d['sha256'] or ep['result']['native_continues']!=0:
            raise ValueError('Unverified transition or continued credit.')
        buffer=ep['onpolicy_buffer']
        if digest(a.development_report.parent/buffer['file'])!=buffer['sha256']:
            raise ValueError('The recorded behavior probability archive changed.')
        with np.load(trail,allow_pickle=False) as z:
            meta=json.loads(str(z['metadata_json']))
            if (meta['trial']!=ep['trial'] or meta['actor'].get('role')!=ROLE or meta['task']['difficulty']!='Normal'
                    or meta['task']['shot']!='Reimu A' or meta['actor']['source_weights_sha256']!=PINNED[WEIGHT]):
                raise ValueError('Original credit metadata does not match its designation.')
        seed=ep['trial']['policy_seed'];name='prospective-'+str(seed);SOURCE[name]=trail
        data,details=dataset(name,contact,relative,None)
        with torch.inference_mode():
            chosen=move.logits(torch.from_numpy(data['x']),torch.from_numpy(data['prior'])).argmax(1).numpy()
        # Every action's target and margin remain available for a downstream supervised experiment.
        target=data['y'][np.arange(len(chosen)),chosen]
        path=a.out/f'development-{seed}.npz';np.savez_compressed(path,**data,conservative_action=chosen,
            privileged_action_for_conservative=target)
        summary['credits'].append(dict(policy_seed=seed,data_file=path.name,derived_sha256=digest(path),
            label='Retrospective teacher action from archived hazard field. Executed actions separately retained.',
            behavior_probabilities=ep['onpolicy_buffer'],**details))
        publish_json(a.out/'manifest.json',summary)
    summary['status']='complete';publish_json(a.out/'manifest.json',summary)


if __name__=='__main__':
    main()
