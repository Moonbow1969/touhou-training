"""Run one zero-Continue, demonstration-conditioned A.U opening swap with an offline verification mode."""
import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if not (REPO/'scripts/train_demo_route.py').is_file():
    REPO = REPO/'work/th14-rl'
sys.path.insert(0, str(REPO))
from scripts import evaluate_credit_story as story
from scripts import evaluate_demo_route_encounters as encounter
from scripts import evaluate_demo_route_stage2_phase as phase2
from scripts import native_story_sensors, train_demo_route
from scripts.checkpoint_storage import digest
from scripts.evaluate_bomb_crossover import DATA, TEMPLATE
from scripts.evaluate_sft_native import output_lock
from scripts.native_story_restart import GameOverMenu
from scripts.train_full_policy import publish_json

HERE=DATA.parent/'th14-training-replay-frontier-261125'
OPENING_SHA='3b3d475440cd029e4315ec3528b7e0d5daa67dc20a2b1718193c53e2f54dbf6f'
REPLAY_SHA='05178d405329a8e8e3bed699307adb8be280c1bb0bf21ecbdf8a22d43ab6e172'
JOIN=656


def splice(actor, opening):
    """Insert a dense prefix, then preserve every original waypoint, key and interpolation after handoff."""
    r=actor.route
    f,xy,keys=[r[k] for k in ('frame_2','positions_2','keys_2')]
    with np.load(opening,allow_pickle=False) as z:
        target,priority=z['target_positions'],z['preferred_keys']
    if target.shape!=(JOIN+5,2) or priority.shape!=(JOIN+5,) or not np.isfinite(target).all():
        raise ValueError('Invalid bounded A.U prefix.')
    next_original=int(f[np.searchsorted(f,JOIN,side='left')])
    t=np.arange(next_original)
    xy_prefix=np.stack([np.interp(t,f,v) for v in xy.T],1).astype(np.float32)
    xy_prefix[:JOIN]=target[:JOIN]
    # The parent indexes the row at the current frame; the 4-column representation is still valid.
    ticks=t[:,None]+np.arange(4)
    original_i=np.clip(np.searchsorted(f,ticks,side='right')-1,0,len(f)-1)
    key_chunk=keys[original_i,np.clip(ticks-f[original_i],0,3)]
    prefix=ticks<496
    key_chunk[prefix]=priority[ticks[prefix]]
    suffix=f>=next_original
    r['frame_2']=np.concatenate((t,f[suffix]))
    r['positions_2']=np.concatenate((xy_prefix,xy[suffix]))
    r['keys_2']=np.concatenate((key_chunk,keys[suffix]))
    # Exhaustively verify original runtime lookup for every tick from handoff until the final waypoint.
    grid=np.arange(JOIN,int(f[-1])+1)
    i=np.clip(np.searchsorted(f,grid,side='right')-1,0,len(f)-1)
    j=np.clip(np.searchsorted(r['frame_2'],grid,side='right')-1,0,len(r['frame_2'])-1)
    assert np.array_equal(keys[i,np.clip(grid-f[i],0,3)],r['keys_2'][j,np.clip(grid-r['frame_2'][j],0,3)])
    errors=[np.max(np.abs(np.interp(grid+4,f,axis)-np.interp(grid+4,r['frame_2'],other)))
            for axis,other in zip(xy.T,r['positions_2'].T)]
    assert max(errors)<1e-5
    return dict(exhaustive_post_return_ticks=len(grid),max_post_return_coordinate_difference=float(max(errors)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=('au','alice','both'),default='au',
                   help='Use the A.U opening, the exact Alice control, or rotate both arms within each seed.')
    p.add_argument('--seeds',nargs='+',type=int,required=True)
    p.add_argument('--out',type=Path)
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--menu-references',nargs=2,type=Path)
    p.add_argument('--menu-crop',action='append',nargs=4,type=int)
    p.add_argument('--resume',action='store_true')
    p.set_defaults(artifact=DATA/'th14-demo-alice-route-261125',
        data=DATA/'sft-native-data-261123/frozen-training-snapshot', shoot_policy=TEMPLATE,
        bomb_model=DATA.parent/'th14-randomized-bomb-value-261125/bomb_value.json',compare_reference=False,
        parakit=Path(phase2.__file__).resolve().parents[2]/'parakit',target='ending',limit=story.LIMITS['ending'],
        start_frame=30,focus_wait=90,game_over_default_reference=None,game_over_retry_reference=None,
        menu_max_error=.035,transition_budget_mib=96,transition_reserve_mib=20480,crossover=[],
        timing_policy=[],restart_hook=None)
    a=p.parse_args()
    if (len(a.seeds)!=len(set(a.seeds)) or min(a.seeds)<0 or not a.verify_only and a.out is None
            or bool(a.menu_references)!=bool(a.menu_crop) or a.menu_crop and len(a.menu_crop)<2):
        p.error('Supply unique seeds, a fresh output, and matched optional menu references and crops.')
    if a.menu_references:
        a.game_over_default_reference,a.game_over_retry_reference=a.menu_references
    torch.set_num_threads(1)
    parent=encounter.make_arms(a)['macro_encounter_clock']
    expected=dict(model_sha256='d659bac965dd558fbaa1dbad0e12f63f5c2b8e0b226c1d42c0942f3d1e2a769b',
        route_sha256='8092e337b03fc6dd43d91b2030db23fbbad9fb3be265f31b1c1e463d3256373c',
        learned_bomb_sha256='51f89c4f816dc89f6bfd0b3e672625b1a4a6c116f0a01cdb3759332cad0b81f8')
    if (any(parent.provenance[k]!=v for k,v in expected.items())
            or parent.provenance['independent_shoot']['weights_sha256']!='d62aea41897019df9bec90f3f3288a350a0dfbd1bd471fbee0377400cdc1e2bc'
            or parent.provenance['independent_shoot']['metadata_sha256']!='ab2eb0fa9bbd9600dfb23198a19e56635c809dd58fc7128ae2864fc44592437f'):
        raise ValueError('The exact baseline component hashes differ from the audited versions.')
    report=json.loads((a.artifact/'report.json').read_text(encoding='utf-8'))
    route=a.artifact/'route.npz'; opening=HERE/'05e0-opening.npz'
    with np.load(route,allow_pickle=False) as z:
        plan,brackets=phase2.stage2_knots(a.data/'th14_ud0613-stage2.npz',report['train_stage_shards']['2'],z['frame_2'])
    freeze=json.loads((HERE/'report.json').read_text(encoding='utf-8'))
    if digest(opening)!=OPENING_SHA or freeze['source']['th14_ud05e0']['original_sha256']!=REPLAY_SHA:
        raise ValueError('A.U opening differs from the audited train source.')
    bomb=json.loads(a.bomb_model.read_text(encoding='utf-8'))
    actor=phase2.ExactStage2Route(a.artifact/'model.pt',route,bomb,parent.policy.shoot_source,plan)
    actor.knots=parent.policy.knots
    handoff=splice(actor,opening)
    provenance=dict(parent.provenance,stage2_phase_knots=plan,stage2_brackets=brackets,
        opening_sha256=OPENING_SHA,opening_replay_sha256=REPLAY_SHA,opening_replay='th14_ud05e0',
        first_30_ticks='inferred from original inputs and calibrated movement; aligned native positions thereafter',
        alternative_through=496,return_complete=JOIN,opening_bomb_policy='retained original learned value head',
        **handoff)
    arms={}
    if a.arm in ('alice','both'):
        control=phase2.ExactStage2Route(a.artifact/'model.pt',route,bomb,parent.policy.shoot_source,plan)
        control.knots=parent.policy.knots
        arms['alice_exact_phase']=replace(parent,policy=control,provenance=dict(parent.provenance,stage2_phase_knots=plan))
    if a.arm in ('au','both'):
        arms['au_opening_exact_phase']=replace(parent,policy=actor,provenance=provenance)
    specification=story.run_specification(a,arms)
    specification['controller'].update(kind='train_demo_au_stage2_opening_with_alice_phase_handoff',
                                       allows_continue=False,runtime_reference_trajectory=True)
    specification['runtime']['runner_sha256']=digest(Path(__file__))
    for module in (encounter,phase2,story,train_demo_route):
        specification['runtime']['source_sha256'][module.__name__]=digest(Path(module.__file__))
    archive=native_story_sensors.configure(a,specification)
    if a.verify_only:
        print(json.dumps(dict(status='verified',**handoff,opening_sha256=OPENING_SHA,
                              specification=specification),default=str))
        return
    if not a.resume:
        a.out.mkdir(parents=True,exist_ok=False)
    menu=GameOverMenu(*a.menu_references,a.menu_crop,.035) if a.menu_references else None
    with output_lock(a.out):
        try:
            story.evaluate(a,arms,specification,a.out/'report.json',menu,None,transition_archive=archive)
        finally:
            path=a.out/'opening-swap-telemetry.json'
            history=json.loads(path.read_text(encoding='utf-8')) if path.is_file() else []
            history.append({k:dict(sessions=v.policy.sessions,stage2=v.policy.stage2_sessions,
                                   encounters=v.policy.later_stage_sessions) for k,v in arms.items()})
            publish_json(path,history)


if __name__=='__main__':
    main()
