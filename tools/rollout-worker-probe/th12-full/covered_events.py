"""Map original live resource transitions to decisions that physically covered the changing tick."""
import hashlib
import json
from pathlib import Path
import numpy as np

STRIDE=10000000

def table(path):
    return np.genfromtxt(path,names=True,delimiter='\t',dtype=None,encoding='utf8',ndmin=1)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def final_input_after_loss(raw,live,d,death):
    """Identify a last input accepted by the game only after a within-tick stock transition."""
    old,lost=live[death-1],live[death];prev,last=d[-2],d[-1]
    if old['tick']!=lost['tick'] or old['stage']!=lost['stage'] or old['spares']!=0 or old['ms']>=lost['ms']:
        raise ValueError('Terminal recovery requires distinct live and dead native samples at the same tick')
    tick=int(lost['tick'])
    if not (last['pre_tick']==last['apply_tick']==prev['post_tick']==tick<last['post_tick']):raise ValueError('Final input does not straddle the separately observed native boundary')
    def native(mask):return sum(bit for i,bit in enumerate((64,128,16,32,8,1,2)) if int(mask)&(1<<i))
    before=native(prev['requested_x11']);after=native(last['requested_x11'])
    if before==after or any((int(x['keys'])&251)!=before for x in (old,lost)):raise ValueError('Cannot identify distinct consumed actions at terminal')
    window=live[(live['stage']==lost['stage'])&(live['tick']>=prev['apply_tick'])&(live['tick']<=last['post_tick'])]
    matching=window[(window['keys']&251)==after]
    if not len(matching) or matching[0]['ms']<=lost['ms'] or matching[0]['spares']!=-1:raise ValueError('Final input was consumed while playable or lacks a native acknowledgement')
    rows=[r.split('\t') for r in (raw/'coverage-decisions.tsv').read_text().splitlines() if r.split('\t')[0]==str(len(d)-1)]
    if len(rows)!=1 or rows[0][2]!='play' or int(rows[0][-1])!=0 or int(rows[0][-2])!=tick:raise ValueError('Missing authentic pre-action live observation')
    return dict(raw_sampled_actions=len(d),live_actions=len(d)-1,first_excluded_action=len(d)-1,
                terminal_credit_action=len(d)-2,pre_stock_record_ms=int(old['ms']),terminal_stock_record_ms=int(lost['ms']),
                first_consumption_ms=int(matching[0]['ms']),first_consumption_tick=int(matching[0]['tick']),
                terminal_tick=tick,consumed_at_death_mask=before,postdeath_consumed_mask=after)

def prepare(raw,fields,row,require_formal=True):
    source=table(raw/'engineering-native.tsv');d=table(raw/'engineering-decisions.tsv')
    path=raw/'coverage-decisions.tsv'
    with path.open() as f:semantic=[s.rstrip('\n').split('\t') for s in f]
    last=semantic[-1];body=semantic[1:-1]
    if last[0]!='#terminal' or last[1] not in ('score_ranking','gameover','gameover_retry') or 'physical_keys=0' not in last or 'stock=-1' not in last:raise ValueError('The true menu, exhausted stock and released physical controls must agree')
    if require_formal and ((raw/'engineering-prefix.tsv').exists() or (raw/'pause-negative.tsv').exists()):raise ValueError('Engineering inputs cannot enter formal PPO data')
    if any(r[2]!='play' and int(r[5])>=0 and r[1]=='loading' for r in body):raise ValueError('Corrected actor released controls during unclassified live gameplay')
    live=source[(source['mode']==0)&(source['stage']==source['manager_stage'])&(source['stage']>=1)&(source['tick']>=0)]
    coords=live['stage'].astype(np.int64)*STRIDE+live['tick']
    if len(live)<2 or tuple(int(live[0][k]) for k in ('stage','spares','bombs'))!=(1,2,2) or live[-1]['spares']!=-1 or np.any(np.diff(coords)<0):raise ValueError('Incomplete original native stock or stage chronology')
    start=d['apply_stage'].astype(np.int64)*STRIDE+d['apply_tick'];end=d['post_stage'].astype(np.int64)*STRIDE+d['post_tick']
    pre=d['pre_stage'].astype(np.int64)*STRIDE+d['pre_tick']
    if require_formal and np.any(pre[1:]!=end[:-1]):raise ValueError('Native tick gap while policy had released control')
    if np.any(np.diff(start)<0) or np.any(end<start):raise ValueError('Unordered native control boundaries')
    death=int(np.flatnonzero(live['spares']==-1)[0]);death_coord=coords[death]
    if not start[-1] <= death_coord <= end[-1]:
        raise ValueError(f'Final stock event not bracketed by the terminal decision: {death_coord}, {start[-1]}, {end[-1]}')
    if np.any(live['spares'][:death]<0) or np.any(live['spares'][death:]!=-1):raise ValueError('Native terminal stock is not final')
    n=len(d);parts=np.zeros((n,5),np.float32);events=[];history=[];final_action=n-1;boundary=None
    # Time survives as the same weak +0.001/s term. Only a genuine applied interval is eligible.
    parts[:,0]=.001*(fields['observed_ns']-fields['onset_ns'])/1e9
    def attribute(j):
        c=coords[j];idx=int(np.searchsorted(start,c,side='left')-1)
        if idx<0 or idx>=n:raise ValueError('Resource transition before first applied action')
        # The previous keys remain physically held between two successive worker decisions.
        if c>end[idx] and (idx+1>=n or c>start[idx+1]):raise ValueError('Reward event occurred after physically supported input')
        return idx
    for j in range(1,death+1):
        a,b=live[j-1],live[j]
        dl=int(b['spares']-a['spares']);db=int(b['bombs']-a['bombs']);ds=int(b['stage']-a['stage'])
        if not (dl or db or ds):continue
        i=attribute(j)
        if dl:
            if b['spares']==-1:
                if dl!=-1:raise ValueError('Unexpected final life differential')
                if i!=n-1:
                    if i!=n-2:raise ValueError('Final life not covered by either boundary action')
                    boundary=final_input_after_loss(raw,live,d,death)
                    final_action=i
                parts[i,4]=-1
            else:parts[i,1]+=dl
            history.append(dict(step=i,event='spares',prior=int(a['spares']),current=int(b['spares'])))
        if db:
            cost=0.
            if not dl and db<0:
                # The game spent actual Bomb stock. It may consume the X edge before a sampled state just after the input release.
                last=int(np.searchsorted(coords,coords[j]-12,side='left'))
                if not np.any((live['keys'][last:j+1]&2)!=0):raise ValueError('Native same-life Bomb stock decrement without observed consumed X')
                cost=.03*db;parts[i,2]+=cost
            history.append(dict(step=i,event='bombs',prior=int(a['bombs']),current=int(b['bombs']),cost=cost))
        if ds:
            if ds!=1:raise ValueError('Unexpected original stage transition')
            parts[i,3]+=2.
        events.append(dict(step=i,stage=int(b['stage']),native_tick=int(b['tick']),spares=int(b['spares']),bombs=int(b['bombs']),spare_delta=dl,bomb_delta=db,stage_delta=ds))
    if parts[final_action,4]!=-1 or parts[:,4].sum()!=-1:raise ValueError('Missing or duplicate terminal reward')
    if boundary:
        eligible=np.ones(n,dtype=np.bool_);eligible[-1]=False
        boundary['native_action_eligible_shape']=[n]
        boundary['native_action_eligible_sha256']=hashlib.sha256(eligible.tobytes()).hexdigest()
        boundary['consumed_terminal_action_start_tick']=int(d[-2]['apply_tick'])
        boundary['consumed_terminal_action_end_tick']=int(d[-2]['post_tick'])
        fields['native_action_eligible']=eligible
        parts[-1]=0
    native=dict(qualified=True,terminal_native_stage=int(live[death]['stage']),death_native_tick=int(live[death]['tick']),
        last_decision_start_tick=int(d[-1]['apply_tick']),last_decision_end_tick=int(d[-1]['post_tick']),death_reward_action=final_action,
        live_interval_gaps=int(np.sum(pre[1:]!=end[:-1])),full_original_actions=n,native_initial=[2,2],last_playable_bombs=int(live[death-1]['bombs']),
        native_terminal_photo_sha256=digest(raw/'terminal-at-control.ppm'),actual_checkpoint_sha256=row['policy_sha256'],
        resource_events=events,raw_sha256={name:digest(raw/name) for name in ('coverage-decisions.tsv','engineering-decisions.tsv','engineering-native.tsv','decisions.tsv','observations.u8','stage-native.tsv','difficulty-native.tsv')})
    if boundary:native['postterminal_input_boundary']=boundary
    fields.update(reward=parts.sum(1),reward_parts=parts)
    row.update(event_history=history,remaining_spare_lives=0,remaining_bombs=int(live[death-1]['bombs']),reward_parts=parts.sum(axis=0).tolist(),covered_native_events=native)
    return native
