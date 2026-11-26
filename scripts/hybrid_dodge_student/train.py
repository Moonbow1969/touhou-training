"""Imitate the hybrid's privileged evasive action choice and motion-verified late stock inputs."""
import argparse
import json
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from .actor import OUT,ROOT,MovementStudent,AnticipatoryContact
from .prepare import WORLD,SOURCE,BASE,CONTACT,dataset
from scripts.checkpoint_storage import digest
from scripts.relative_phase_student import RelativeActor
from scripts.evaluate_stage_policies import read_policy

TRAIN=('late5','late6','practice4','2201','2202')
DEV=('2402',)
CHECK=('held4','held5','2601','3301','3601')
SEED=71391


def summary(data,mask,chosen):
    if not np.any(mask):
        return {'n':0}
    m,prior,action=data['m'][mask],data['prior'][mask],chosen[mask]
    row=np.arange(len(prior));y=data['y'][mask][row,prior]
    was=m[row,prior];now=m[row,action]
    quiet=was>=12
    danger=was<4
    onset=data['hit'][mask]
    exp=data['expert'][mask];actual=data['actual'][mask]
    return dict(n=len(row),prior_negative=int(sum(was<0)),student_negative=int(sum(now<0)),
        teacher_negative=int(sum(m[row,y]<0)),prior_danger=int(sum(danger)),move_changes=int(sum(action!=prior)),
        quiet_rows=int(sum(quiet)),quiet_changes=int(sum(quiet&(action!=prior))),
        dangerous_positive_margin_changes=int(sum(danger&(now>was+.75))),
        dangerous_worse_margin_changes=int(sum(danger&(now<was-.75))),
        teacher_exact_on_intervention=int(sum((y!=prior)&(action==y))),teacher_interventions=int(sum(y!=prior)),
        teacher_exact_overall=float(np.mean(y==action)),onset_rows=int(sum(onset)),
        onsets_base_equals_behavior=int(sum(onset&(prior==actual))),onsets_change=int(sum(onset&(action!=prior))),
        onsets_projected_gain=int(sum(onset&(now>was+.75))),onsets_projected_loss=int(sum(onset&(now<was-.75))),
        measured_motion_rows=int(sum(exp)),prior_motion_agreement=int(sum(exp&(prior==actual))),
        student_motion_agreement=int(sum(exp&(action==actual))),all_four_base_behavior_match_fraction=(float(data['measured'][mask].mean()) if np.any(data['phase'][mask]!=-999) else None))


@torch.inference_mode()
def evaluate(model,data):
    pred=model.logits(torch.from_numpy(data['x']),torch.from_numpy(data['prior'])).argmax(1).numpy()
    masks={'all':np.ones(len(pred),bool),'stage2_first500':(data['stage']==2)&(data['frame']<=500)}
    for stage in np.unique(data['stage']):
        masks['stage'+str(stage)]=data['stage']==stage
    for spell in np.unique(data['phase']):
        if spell>=0:masks['spell_'+str(spell)]=data['phase']==spell
    return {k:summary(data,m,pred) for k,m in masks.items() if m.any()}


def objective(model,data,ix,rng):
    tensors=[torch.from_numpy(data[k][ix]) for k in ('x','m','y','prior','actual','expert')]
    x,m,y,base,actual,expert=tensors
    keep,alt,margin=model.encode(x)
    ids=torch.arange(18)[None,:]
    change=y!=ids
    # Direct supervised choice for all counterfactual prior commands, with scarce teacher switches upweighted.
    keep_loss=F.binary_cross_entropy_with_logits(-keep,change.float(),pos_weight=torch.tensor(2.))
    row,p=torch.where(change)
    if len(row):
        dest=alt[row,p//9]
        # Near-tied routes get partial credit. Preserve the actual teacher argmax by including hard CE.
        penalty=(m[row].max(1,keepdim=True).values-m[row]).clamp(0,30)
        soft=(-penalty/2).softmax(1)
        alt_loss=.5*F.cross_entropy(dest,y[row,p])-.5*(soft*dest.log_softmax(1)).sum(1).mean()
    else:alt_loss=alt.sum()*0
    margin_loss=F.smooth_l1_loss(margin,(m/12).clamp(-1,2))
    if expert.any():
        action_loss=F.nll_loss(model.logits(x[expert],base[expert]),actual[expert])
    else:action_loss=x.sum()*0
    return keep_loss+alt_loss+.35*margin_loss+.30*action_loss


def train():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'student.pt').exists() or (OUT/'conservative.pt').exists():
        raise FileExistsError('Use a fresh artifact directory or reproduce in a separate workspace.')
    started=perf_counter(); torch.manual_seed(SEED);torch.set_num_threads(2);rng=np.random.default_rng(SEED)
    if digest(BASE)!='d5bfabb005ee68f2bcc2410e6649f9df528262d28c3f215a35b2c38fbc3acdde':
        raise ValueError('Wrong pure student.')
    base_state=torch.load(BASE,map_location='cpu',weights_only=True)['state_dict']
    relative=RelativeActor().eval();relative.load_state_dict(base_state)
    contact=AnticipatoryContact().eval();contact_state=torch.load(CONTACT,map_location='cpu',weights_only=True)['state_dict']
    contact.load_state_dict(contact_state)
    sft=read_policy(WORLD/'overnight-title-equal-61/sft/policy.pt','cpu',True)
    metadata={};datasets={}
    def load(names):
        for name in names:
            print('Preparing',name,flush=True)
            datasets[name],metadata[name]=dataset(name,contact,relative,sft)
    load(TRAIN+DEV)
    all_train={k:np.concatenate([datasets[s][k] for s in TRAIN]) for k in datasets[TRAIN[0]]}
    x,m,y=all_train['x'],all_train['m'],all_train['y']
    challenge=np.flatnonzero((m.min(1)<4)&(m.max(1)>m.min(1)+1))
    expert=np.flatnonzero(all_train['expert'])
    optmodel=MovementStudent();optim=torch.optim.AdamW(optmodel.parameters(),lr=.0015,weight_decay=.005)
    logs=[]
    for epoch in range(20):
        losses=[]
        for _ in range(34):
            ix=np.r_[rng.integers(len(x),size=96),rng.choice(challenge,size=96),rng.choice(expert,size=64)]
            loss=objective(optmodel,all_train,ix,rng)
            optim.zero_grad(set_to_none=True);loss.backward();optim.step();losses.append(loss.item())
        logs.append(float(np.mean(losses)));print('Epoch',epoch+1,round(logs[-1],4),flush=True)
    # Offset calibrated on one older whole credit. Never use later native credits or late observer holdouts.
    dev=datasets['2402'];cal=[]
    for offset in (-2.,-1.,0.,1.,2.):
        optmodel.keep_offset.fill_(offset)
        result=evaluate(optmodel,dev)['all']
        drift=result['quiet_changes']/max(1,result['quiet_rows'])
        cal.append(dict(offset=offset,quiet_drift=drift,negative=result['student_negative'],
                        danger_worse=result['dangerous_worse_margin_changes'],baseline_negative=result['prior_negative']))
    admissible=[v for v in cal if v['quiet_drift']<=.025]
    winner=min(admissible or cal,key=lambda v:(v['negative'],v['danger_worse'],-v['offset']))
    optmodel.keep_offset.fill_(winner['offset']);optmodel.eval()
    weights=OUT/'student.pt'
    torch.save(dict(schema='th14-hybrid-privileged-state-only-v1',state_dict=base_state,contact=contact_state,
                    movement=optmodel.state_dict()),weights)
    # Freeze weights before loading independent late snippets and native test credits.
    fingerprint=digest(weights)
    load(CHECK)
    measurements={name:evaluate(optmodel,data) for name,data in datasets.items()}
    if digest(weights)!=fingerprint:
        raise RuntimeError('Frozen test weights changed.')
    report=dict(schema='th14-hybrid-privileged-state-only-v1',sha256=fingerprint,seed=SEED,
        train=TRAIN,calibration=DEV,check=CHECK,sources=metadata,epochs=logs,offset_candidates=cal,
        selected=winner,measurements=measurements,base_sha256=digest(BASE),contact_sha256=digest(CONTACT),
        base_role='Frozen relative-clock student including its original Shoot/Bomb. Pure learned four heads with a learned movement/focus replacement.',
        contact_role='Frozen 12-tick per-action bullet network, all 96 raw retained entities. Prior training included official human train replays and used human validation for epoch selection.',
        historical_teacher='Counterfactual original suggest_dodge, player radius 1 and 12 ticks. Late inputs include recorded combined projectile/body/straight-laser action margins. Stage1/2 labels use constant-velocity projectile fields with stored bodies/lasers optionally added only when an entity family is complete.',
        actual_bc='Stable four-tick executed movement/focus and endpoint within 0.51 native units, no Bomb, same life. Prior reconstructed from original hybrid SFT on late native observation history; identity of every historical observer policy is not recorded.',
        scored='Negative margin is a projected surface crossing on the archived field. Onset counts use actual state 1 to 4; different actions do not create observed outcomes.',
        seconds=perf_counter()-started)
    (OUT/'training.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    for name in DEV+CHECK:
        r=measurements[name]['all'];print(name,{k:r[k] for k in ('n','prior_negative','student_negative','quiet_changes','quiet_rows','measured_motion_rows','prior_motion_agreement','student_motion_agreement')})
    print('Frozen',fingerprint,flush=True)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    train()
