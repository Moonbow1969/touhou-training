"""Independent full-credit native cadence, physical input, and camera-likelihood verification."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch

from summarize import summarize
from difficulty import verify_difficulty

def one(raw, source, output, ranking_reference):
    site=raw.parent.name
    output.mkdir(exist_ok=True,parents=True)
    proof=summarize(raw,output/"verified",ranking_reference=ranking_reference)
    rank=verify_difficulty(raw,2,proof['seconds'],'9fa30534d6710abd245dfa64fc222ea6816f960b96974fed536a00ae1c08094b')
    assert proof['mode']=='engineering' and proof['policy_sha256']==hashlib.sha256(source.read_bytes()).hexdigest() and proof['native_outcome']=='game_over'
    get=lambda n:np.genfromtxt(raw/n,names=True,delimiter='\t',dtype=None,encoding='utf8',ndmin=1)
    d=get('engineering-decisions.tsv');s=get('engineering-native.tsv');t=get('decisions.tsv');n=len(d)
    assert n==proof['actions'] and np.all(d['pre_tick']%4==0) and np.array_equal(d['action'],np.arange(n))
    assert np.array_equal(d['requested_x11'],d['observed_x11']) and np.array_equal(d['requested_x11'],t['keys'])
    live=d['pre_stage'][:-1]==d['pre_stage'][1:];gaps=np.diff(d['pre_tick'])[live]
    assert np.all(gaps>0) and np.all(gaps%4==0)
    with np.load(output/'verified'/proof['trajectory']) as z:
        camera=torch.from_numpy(z['frames'].copy()).float()/255
        diff=torch.zeros_like(camera);diff[1:]=(camera[1:]-camera[:-1]).abs();x=torch.stack((camera,diff),1)
        model=torch.jit.load(str(source),map_location='cpu').eval()
        torch.set_num_threads(2);scores=[]
        with torch.no_grad():
            for i in range(0,n,48):
                heads=model(x[i:i+48]);scores.append(torch.cat(heads[:4],1).numpy())
        preds=np.concatenate(scores)
        dl=float(np.max(np.abs(preds-z['camera_logits'])))
        lps=[]
        for j,h in enumerate(np.split(preds,[9,11,13],axis=1)):
            logp=torch.from_numpy(h).log_softmax(-1).numpy();lps.append(logp[np.arange(n),z['action'][:,j]])
        dlp=float(np.max(np.abs(np.stack(lps,1)-z['logp'])))
        assert dl<.0004 and dlp<.0004 and np.all(z['value']==0) and np.array_equal(z['keys'],z['observed_keys'])
    # Map actual X masks to the original game's seven consumed input flags.
    xb=d['requested_x11'].astype(int)
    native_expected=sum(((xb>>bit)&1)*code for bit,code in enumerate((64,128,16,32,8,1,2)))
    raw_mask=251
    rows=[];all_bits=[];near=[]
    for i in range(n):
        if d['apply_stage'][i]!=d['post_stage'][i]:continue
        mask=(s['stage']==d['apply_stage'][i])&(s['manager_stage']==s['stage'])&(s['mode']==0)&(s['tick']>d['apply_tick'][i])&(s['tick']<=d['post_tick'][i])
        if not np.any(mask):continue
        source_rows=s[mask]
        last=int(source_rows['keys'][-1]) & raw_mask
        expected=int(native_expected[i]);rows.append((i,last,expected));all_bits.extend([(int(k)&raw_mask,expected) for k in source_rows['keys']])
        near.append(bool(np.any((source_rows['keys']&raw_mask)==expected)))
    observed=np.array(rows);middle=np.array(all_bits)
    focus=[dict(requested=state,actions=int(np.count_nonzero(((observed[:,2]&8)>0)==state)),
                native_agreement=float(np.mean(((observed[:,1]&8)>0)[((observed[:,2]&8)>0)==state]==state)))for state in (False,True)]
    tags={n:hashlib.sha256((raw/n).read_bytes()).hexdigest()for n in ('engineering-native.tsv','engineering-decisions.tsv','decisions.tsv','terminal-at-control.ppm','difficulty-start-hud.png','observations.u8')}
    out=dict(scope='Engineering only, no TRAIN/HELD budget. Input actions came from the frozen four-head human trial policy. Raw game consumed key mask uses original 0.02a 0x4d2670 (mask 251).',host=site,seed=proof['seed'],policy_sha256=proof['policy_sha256'],actions=n,
        stage2=proof['stage2_reached'],native_terminal=proof['native_terminal_state'],terminal_spares=proof['remaining_spare_lives'],terminal_bombs=proof['remaining_bombs'],
        focus_controls=focus,post_action_eligible=len(rows),native_all_key_agreement_end_of_hold=float(np.mean(observed[:,1]==observed[:,2])),
        agreement_at_any_observed_tick_in_hold=float(np.mean(near)),native_input_samples=len(middle),all_physical_X_states_equal=bool(np.all(d['observed_x11']==d['requested_x11'])),
        native_input_agreement_per_bit={str(k):float(np.mean((observed[:,1]&k)==(observed[:,2]&k)))for k in (64,128,16,32,8,1,2)},
        steps_per_second=n/proof['seconds'],native_tick_per_second=float(gaps.sum()/((t['onset_ns'][-1]-t['onset_ns'][0])/1.e9)),seconds=proof['seconds'],
        cadence_native_tick_distribution={str(int(k)):int((gaps==k).sum()) for k in np.unique(gaps)},exact_four_fraction=float(np.mean(gaps==4)),
        missed_four_tick_slots=int(sum(gaps//4-1)),duplicated_or_reversed=0,frame_bracket_rejections=int(d['capture_retries'].sum()),stalled_poll_checks=int(d['frozen_checks'].sum()),
        longest_decision_wall_ms=float(np.diff(t['onset_ns']).max()/1e6),definition='stalled_poll_checks counts each 125ms diagnostic UI check while waiting; longest inter-decision occupancy includes original loading. Exact stride uses consecutive model observation ticks within the same original stage.',
        maximum_second_host_logits_error=dl,maximum_second_host_sampled_logp_error=dlp,raw_sha256=tags,trajectory_sha256=proof['trajectory_sha256'],rank_proof=rank)
    target=output/f'{site}-engineering-native-receipt.json'
    target.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ('host','actions','cadence_native_tick_distribution','native_all_key_agreement_end_of_hold','maximum_second_host_sampled_logp_error')},indent=2))
    return out

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw',type=Path)
    parser.add_argument('model',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--ranking-reference',type=Path)
    args=parser.parse_args()
    one(args.raw,args.model,args.output,args.ranking_reference)
