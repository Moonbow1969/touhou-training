"""Verify a frozen pure neural challenger or score one ending-target stock credit with native resources."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .actor import OUT,ROOT,REPO,StockHybridStudent
from scripts import evaluate_credit_story as story
from scripts import native_story_sensors,relative_phase_student,state_route_student
from scripts.checkpoint_storage import digest
from scripts.evaluate_sft_native import Arm,output_lock
from scripts.native_story_restart import GameOverMenu
from scripts.train_full_policy import publish_json

WEIGHT=OUT/'conservative.pt'
PINNED={WEIGHT:'3cdd6a1501196dec9a46f07c5d708a6c0130438c7a24ed99c2e67585ee68934b',
    Path(__file__).with_name('actor.py'):'52184b0d420b5c8073ce79de72c1feb31d453de5837550edd59436c557aa6eaf',
    Path(relative_phase_student.__file__):'f65ed3bd91e45abf0bfa1e1f05f60cb01c5ffc5562c73a9c72dfc79fe3a3c586',
    Path(state_route_student.__file__):'705bf0b51b09684d1b97d32519d80df31453b849394d9aa64dcb5a88e09f972f',
    ROOT/'work/th14-menu-default-runtime-261125.png':'1684aeb1fea7f3c39d181c6dcd1a9adfd7b063f6b892aac901a75a4cac2442c2',
    ROOT/'work/th14-menu-retry-runtime-261125.png':'7c2ca1a1ec29d46bb1088ec86e320d06a52ef05f619b25e920ff39c6709afbfe'}


def resources(path):
    state=json.loads(path.read_text(encoding='utf-8-sig'))
    if state['status']!='complete' or len(state['completed'])!=1:
        raise ValueError('Expected one completed credit before resource auditing.')
    record=state['completed'][0];r=record['result'];shard=r['transition_shard'];data=path.parent/shard['path']
    if digest(data)!=shard['sha256'] or r['native_continues']!=0:
        raise ValueError('Native replay attribution or zero Continue requirement failed.')
    result=[]
    with np.load(data,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata_json']));col={v:i for i,v in enumerate(meta['native_field_order'])}
        before,after=z['native_before'],z['native_after'];sources=before[:,col['stage']]
        misses=z['native_life_losses'];uses=z['actual_bombs_consumed'];actions=z['action']
        if int(misses.sum())!=r['deaths'] or int(uses.sum())!=r['actual_bombs_consumed'] or len(actions)!=r['decision_count']:
            raise ValueError('Credit archive resource totals diverge from its game report.')
        for s in r['stage_records']:
            mask=sources==s['stage'];n_lives=int(misses[mask].sum());n_bombs=int(uses[mask].sum())
            if (n_lives,n_bombs)!=(s['lives_lost'],s['bombs_consumed']):
                raise ValueError('Stage-aligned resource accounting diverges from the archive.')
            onsets=mask&(before[:,col['player_state']]==1)&(after[:,col['player_state']]==4)
            result.append(dict(stage=s['stage'],decisions=int(mask.sum()),entry=s['entry']['resources'],exit=s['exit']['resources'],
                terminal_reason=s['exit']['reason'],last_controlled_frame=s['exit']['last_controlled_frame'],
                first_observed_contact_frame=int(after[onsets,col['stage_frame']][0]) if onsets.any() else None,
                observed_contacts=int(onsets.sum()),actual_life_losses=n_lives,actual_bombs_consumed=n_bombs,
                bomb_command_decisions=int(actions[mask,3].sum())))
    return dict(outcome=r['outcome'],trial=record['trial'],target=state['specification']['task']['target'],
        one_credit_clear=r['one_credit_clear'],native_continues=r['native_continues'],terminal=r['native_terminal'],
        decision_count=r['decision_count'],life_losses=r['deaths'],actual_bombs_consumed=r['actual_bombs_consumed'],
        stage_resources=result,archive_sha256=shard['sha256'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=('challenger','anchor'),default='challenger')
    p.add_argument('--ppo-checkpoint',type=Path,
                   help='Evaluate a published joint-movement PPO descendant from a finished development run.')
    p.add_argument('--seed',type=int,default=4001)
    p.add_argument('--out',type=Path,help='Fresh evaluation output directory.')
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--audit-report',type=Path,help='Audit one existing completed report offline and print resources.')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.audit_report:
        print(json.dumps(resources(a.audit_report)));return
    if a.seed<0 or (not a.verify_only and a.out is None):
        p.error('Supply a nonnegative seed and a fresh output directory for native scoring.')
    torch.set_num_threads(1)
    for path,reference in PINNED.items():
        if digest(path)!=reference:
            raise ValueError('Frozen file hash mismatch: '+path.name)
    gate=json.loads((OUT/'frozen-checks.json').read_text(encoding='utf-8'))
    if not gate['final_credit_gate'] or gate['weights_sha256']!=PINNED[WEIGHT]:
        raise ValueError('A matching frozen holdout certificate is required.')
    chosen=WEIGHT
    if a.ppo_checkpoint:
        if a.arm!='challenger':
            p.error('A PPO checkpoint is a challenger arm.')
        parent=json.loads(a.ppo_checkpoint.with_name('report.json').read_text(encoding='utf-8'))
        if (parent.get('status')!='complete' or parent['specification']['development']['role']!='development_on_policy_joint_movement'
                or len(parent.get('updates',[]))!=1
                or parent['updates'][0]['candidate']['sha256']!=digest(a.ppo_checkpoint)
                or parent['specification']['policies']['development_joint_ppo']['source_weights_sha256']!=PINNED[WEIGHT]):
            raise ValueError('PPO descendant is not attributed to a completed compatible development run.')
        chosen=a.ppo_checkpoint
    a.seeds=[a.seed];a.parakit=REPO.parent/'parakit';a.start_frame=30;a.focus_wait=90
    a.target='ending';a.limit=story.LIMITS[a.target];a.menu_max_error=.035
    a.game_over_default_reference=ROOT/'work/th14-menu-default-runtime-261125.png'
    a.game_over_retry_reference=ROOT/'work/th14-menu-retry-runtime-261125.png'
    a.menu_crop=[[39,256,48,20],[39,356,118,20],[15,207,70,24]]
    a.restart_hook=None;a.crossover=[];a.timing_policy=[]
    a.transition_budget_mib=96;a.transition_reserve_mib=20480
    if a.out is None:a.out=OUT/'unused-verify-path'
    if a.arm=='challenger':
        actor=StockHybridStudent(chosen);name='joint_ppo_challenger' if a.ppo_checkpoint else 'hybrid_distilled_state_only'
    else:
        actor=relative_phase_student.StockRelativeActor(WEIGHT);name='frozen_relative_anchor'
    arms={name:Arm(actor,False,dict(arm=a.arm,weights_sha256=digest(chosen),
           base_weights_sha256='d5bfabb005ee68f2bcc2410e6649f9df528262d28c3f215a35b2c38fbc3acdde',
           raw_projectile_input_slots=96 if a.arm=='challenger' else 3,
           training='privileged counterfactual dodge labels and verified late native controller inputs' if a.arm=='challenger' else 'relative-clock route student',
           runtime_reference_trajectory=False,runtime_clearance_solver=False))}
    specification=story.run_specification(a,arms)
    specification['runtime']['runner_sha256']=digest(Path(__file__))
    specification['runtime']['source_sha256'].update({path.name:expected for path,expected in PINNED.items() if path.suffix=='.py'})
    specification['controller']['kind']='state-only neural four-head action policy'
    archive=native_story_sensors.configure(a,specification)
    if a.verify_only:
        print(json.dumps(dict(status='verified',specification=specification,verified_files={p.name:h for p,h in PINNED.items()})));return
    if not a.resume:a.out.mkdir(parents=True,exist_ok=False)
    menu=GameOverMenu(a.game_over_default_reference,a.game_over_retry_reference,a.menu_crop,.035)
    with output_lock(a.out):
        report=a.out/'report.json'
        story.evaluate(a,arms,specification,report,menu,None,transition_archive=archive)
        if json.loads(report.read_text(encoding='utf-8'))['status']=='complete':
            result=resources(report);publish_json(a.out/'resource-audit.json',result);print(json.dumps(result))


if __name__=='__main__':
    main()
