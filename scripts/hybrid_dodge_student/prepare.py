"""Build training and measurement rows without access to a game process."""
import json
from pathlib import Path

import numpy as np
import torch

from .actor import OUT,ROOT,REPO,feature_vectors
from scripts.checkpoint_storage import digest
from scripts.relative_phase_student import RelativeActor,EventClock
from scripts.state_route_student import features
from scripts.pretrain_policy import condition_context,append_motion_history,append_stage_context
from scripts.stage_risk import body_clearances,laser_clearances

WORLD=ROOT/'outputs/touhou-sft-world'
BASE=ROOT/'outputs/th14-relative-phase-student-261125/student.pt'
CONTACT=ROOT/'outputs/th14-anticipatory-contact-12tick-261125/model.pt'
SOURCE={
 'late5':WORLD/'overnight-stage5-onpolicy/th14-native-stage5.npz',
 'late6':WORLD/'overnight-stage6-onpolicy/th14-native-stage6.npz',
 'practice4':WORLD/'stage4-practice-risk-capture/th14-native-stage4.npz',
 'held4':WORLD/'overnight-title-equal-onpolicy/th14-native-stage4.npz',
 'held5':WORLD/'overnight-title-equal-onpolicy/th14-native-stage5.npz',
}
for name,key in [('2201','th14-demo-alice-phase-2201'),('2202','th14-demo-alice-phase-2202'),
                 ('2402','th14-demo-exact-stage2-2402'),('2601','th14-relative-student-ending-2601'),
                 ('3301','th14-anticipatory-64-3301'),('3601','th14-relative-full-eval-online_ppo-3601')]:
    SOURCE[name]=next(WORLD.glob(key+'*/transition-shards/*.npz'),None)


def direction(keys):
    x=np.where(keys&64,-1,(keys&128)>0)
    y=np.where(keys&32,1,-((keys&16)>0).astype(int))
    return (1+y)*3+x+1+9*((keys&8)>0)


def clearances(b,c):
    from scripts.stage_risk import _DIRECTIONS
    all_m=[]
    for i in range(0,len(b),40):
        bullet=torch.from_numpy(b[i:i+40])
        ctx=torch.from_numpy(c[i:i+40])
        dt=torch.arange(1,13)[None,None,None,:,None]
        p0=ctx[:,:2]*torch.tensor([192.,448.])
        v=torch.from_numpy(_DIRECTIONS)[None]*torch.tensor([4.5,2.])[:,None,None]
        traj=(p0[:,None,None,None,:]+v.reshape(1,18,1,1,2)*dt).clamp(torch.tensor([-184.,32.]),torch.tensor([184.,432.]))
        active=(bullet[:,:,6]>0)&(bullet[:,:,5]>0)&(bullet[:,:,4]>0)
        bx=bullet[:,:,None,:2]*torch.tensor([384.,448.])+bullet[:,:,None,2:4]*16*dt[:,0]
        diff=bx[:,None]-(traj-p0[:,None,None,None,:])
        distances=torch.linalg.vector_norm(diff,dim=-1)-1-bullet[:,None,:,None,4]*32
        all_m.append(distances.masked_fill(~active[:,None,:,None],512).amin((2,3)).numpy())
    return np.concatenate(all_m)


def teacher_actions(m):
    """Same 12-tick clearance and focus preference as the original hybrid suggest_dodge."""
    ids=np.arange(18)
    preference=m[:,None,:]+.5*(ids[None,:,None]//9==ids[None,None,:]//9)
    alt=preference.argmax(2)
    val=m[np.arange(len(m))[:,None],alt]
    change=(m<4)&(val>m+.75)
    return np.where(change,alt,ids[None]).astype(np.int64)


@torch.inference_mode()
def relative_actions(model,before,columns,b,c):
    clock=EventClock(); physical=[];et=[];stock=[];hp=[]
    for j,row in enumerate(before):
        stage,frame,size,spell,state,previous,bombs=[row[columns[k]] for k in
            ('stage','stage_frame','boss_max_hp','spell_id','player_state','previous_x','bombs')]
        size=max(0,size)
        e=clock.update(stage,frame,size,spell,c[j,8]>0)
        x,_,raw=features(c[j],b[j],stage,frame,size,spell,row[columns['boss_vulnerable']]>0,e[0],state,previous,bombs)
        physical.append(x);et.append(e);stock.append(raw);hp.append(np.clip(row[columns['boss_hp']]/max(1,size),0,1))
    parts=[torch.from_numpy(np.asarray(x,np.float32)) for x in (physical,et,hp,stock)]
    batches=[]
    for i in range(0,len(b),512):
        j,z,bomb,*_=model(*(v[i:i+512] for v in parts))
        batches.append(np.column_stack([j.argmax(1),z.argmax(1),bomb.argmax(1)]))
    return np.concatenate(batches)


def override_sensor_labels(z,idx,b,c,m):
    extra=0;valid_rows=np.zeros(len(m),bool)
    if 'sensor_decision_index' not in z:
        return extra,valid_rows
    decision=z['sensor_decision_index']; stat=z['sensor_status']
    mapping={v:i for i,v in enumerate(idx)}
    for name,col,fn in [('bodies',1,body_clearances),('straight_lasers',2,laser_clearances)]:
        key='sensor_'+name
        arr,ind,source=z[key],z[key+'_indptr'],z[key+'_source_count']
        for row,original in enumerate(decision):
            if original not in mapping or stat[row,col]!=1:
                continue
            i=mapping[original]
            if source[row]==ind[row+1]-ind[row]:
                # A source is certified complete for this recorded family; the other family may still be absent.
                valid_rows[i]=True
                entities=arr[ind[row]:ind[row+1]]
                if len(entities):
                    v=fn({'bullets':b[i],'context':c[i]},1.,entities).reshape(-1)
                    extra+=int(np.any(v<m[i]));m[i]=np.minimum(m[i],v)
    return extra,valid_rows


@torch.inference_mode()
def dataset(name,contact,relative,sft):
    path=SOURCE[name]
    if path is None or not path.is_file():
        raise FileNotFoundError('The private source for dataset '+name+' is unavailable.')
    with np.load(path,allow_pickle=False) as z:
        b=z['bullets']; c=z['context'][:,:10].copy()
        native='native_before' in z
        if native:
            meta=json.loads(str(z['metadata_json']));col={k:i for i,k in enumerate(meta['native_field_order'])}
            before=z['native_before'];after=z['native_after'];actual=z['action'];state=before[:,col['player_state']]
            frame=before[:,col['stage_frame']];stage=before[:,col['stage']]
            prior_all=relative_actions(relative,before,col,b,c)
            first_event=(state==1)&(after[:,col['player_state']]==4)
            valid=(state==1)&z['four_tick_stage_transition']&(before[:,col['previous_x']]==0)&(actual[:,3]==0)
            idx=np.flatnonzero(valid)
            stage,frame,prior=stage[idx],frame[idx],prior_all[idx,0]
            phase=before[idx,col['spell_id']]
            actual_joint=(actual[:,0]+9*actual[:,1])[idx]
            actual_four_match=(actual[idx,0]+9*actual[idx,1]==prior)&(actual[idx,2]==prior_all[idx,1])&(actual[idx,3]==prior_all[idx,2])
            hit=first_event[idx];expert=np.zeros(len(idx),bool)
            b,c=b[idx],c[idx]
            m=clearances(b,c); extra,scored_sensors=override_sensor_labels(z,idx,b,c,m)
        else:
            meta=json.loads(path.with_suffix('.json').read_text())
            if meta['native_profile']!={'difficulty':1,'character':0,'subshot':0}:
                raise ValueError('Expected Normal Reimu A.')
            frame=z['stage_frame'];stage=np.full(len(b),meta['stage'])
            ctx=append_motion_history(condition_context(c,1,'ReimuA'),frame,sft.motion_spans,sft.retain_recent_motion)
            if sft.stage_clock:ctx=append_stage_context(ctx,frame,meta['stage'])
            chosen=np.concatenate([sft.predict({'bullets':b[i:i+192],'context':ctx[i:i+192]},deterministic=True)[0]
                                   for i in range(0,len(b),192)])
            prior=chosen[:,0]+9*chosen[:,1]
            words=z['expert_action_chunk'];actions=direction(words)
            actual_joint=actions[:,0];phase=np.full(len(b),-999);hit=np.zeros(len(b),bool)
            actual_four_match=np.zeros(len(b),bool)
            nativepos=c[:,:2]*[192.,448.]
            route=np.load(WORLD/'th14-demo-alice-route-261125/route.npz')
            deltas=route['action_deltas']/4;bounds=route['bounds'];p=nativepos.copy()
            for k in actions.T:p=np.clip(p+deltas[k],*bounds)
            expert=np.r_[((np.diff(frame)==4)&(np.diff(c[:,3])==0)&(np.linalg.norm(p[:-1]-nativepos[1:],axis=1)<=.51)),False]
            expert&=(actions==actions[:,:1]).all(1)&((words&2)==0).all(1)
            m=z['action_clearances'].min(1).reshape(-1,18)
            scored_sensors=np.ones(len(b),bool);extra=None
        vec=torch.cat([feature_vectors(contact,torch.from_numpy(b[i:i+128]),torch.from_numpy(c[i:i+128]),
                                       torch.from_numpy(stage[i:i+128].astype('int64'))) for i in range(0,len(b),128)]).numpy()
    return dict(x=vec,m=m,y=teacher_actions(m),prior=prior.astype('int64'),actual=actual_joint.astype('int64'),
        expert=expert,hit=hit,frame=frame,stage=stage,phase=phase,boss=c[:,8]>0,measured=actual_four_match,
        supplemental=scored_sensors),dict(file=str(path.relative_to(WORLD)) if path.is_relative_to(WORLD) else path.name,sha256=digest(path),rows=len(c),
        stage_counts={str(k):int(np.sum(stage==k)) for k in np.unique(stage)},
        genuine_motion_aligned=int(expert.sum()),supplemental_known_any_family=int(scored_sensors.sum()),
        supplemental_active_changes=extra,native_phase_ids=sorted(set(map(int,phase))),native_contact_onsets=int(hit.sum()))
