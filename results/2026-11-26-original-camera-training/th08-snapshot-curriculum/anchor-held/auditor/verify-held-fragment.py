"""Verify a held TH08 continuation, with the original prefix and the arm's actual suffix policy."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import time
import zipfile
import numpy as np
import torch

def digest(blob):return hashlib.sha256(blob).hexdigest()
def sha(p):return digest(p.read_bytes())
def write(p,row):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.pending')
    tmp.write_text(json.dumps(row,indent=2)+'\n');tmp.replace(p)
def npz(b):
    with np.load(io.BytesIO(b),allow_pickle=False)as ar:return {k:ar[k]for k in ar.files}
def guest_npz(arrays):
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_DEFLATED)as target:
        for k,v in arrays.items():
            info=zipfile.ZipInfo(k+'.npy');info.create_system=3;info.compress_type=zipfile.ZIP_DEFLATED
            with target.open(info,'w',force_zip64=True)as entry:np.lib.format.write_array(entry,v,allow_pickle=False)
    return buf.getvalue()
def witness_hud(proof,read,tr,last):
    h=np.frombuffer(read('stock-rgb.ubyte'),np.uint8).reshape(proof['stock']['shape'])
    assert h.shape[1:]==(29,125,3)
    keys=[]
    for cy,color in ((6,'life'),(22,'bomb')):
        active=[]
        for cx in range(6,119,16):
            r,g,b=np.moveaxis(h[:,cy-6:cy+7,cx-6:cx+7].astype(np.int16),-1,0)
            cell=((r>125)&(r-g>32)&(r-b>36)) if color=='life' else ((b>110)&(b-r>20)&(b-g>28))
            active.append(cell.sum((1,2))>=16)
        cells=np.array(active).T
        count=cells.sum(1);regular=np.all(cells==(np.arange(cells.shape[1])[None]<count[:,None]),axis=1)
        keys.append(np.where(regular,count,-1))
    assert np.array_equal(keys[0],np.r_[tr['spares'],tr['next_spares'][-1]]) and np.array_equal(keys[1],np.r_[tr['bombs'],tr['next_bombs'][-1]])
    bits=np.frombuffer(read('rank-rgb.ubyte'),np.uint8).reshape(proof['normal_rank']['shape']).astype(np.int16)
    r,g,b=np.moveaxis(bits,-1,0);observed=(b>85)&(b-r>35)&(b-g>18)
    overlap=(observed&rank_template).sum((1,2))/np.maximum((observed|rank_template).sum((1,2)),1)
    return int((overlap[:-1]>.70).sum())

def accept(seed,compact,shard):
    with tarfile.open(compact)as z:
        files={m.name:m for m in z.getmembers() if m.isfile()};read=lambda n:z.extractfile(files[n]).read()
        seal=json.loads(read('compact-seal.json'))
        for name,expected in seal['source_sha256'].items():assert digest(read(name))==expected,name
        src=json.loads(read('receipt.json'));physical=json.loads(read('fragment-receipt.json'));pre=json.loads(read('vm-policy-anchor.json'));anchor=json.loads(read('vm-anchor.json'))
        spec=json.loads(read('branch-release.json'));binding=json.loads(read('branch-provenance.json'));assert json.loads(read('branch-protocol.json'))==reg
        assert digest(read('branch-protocol.json'))==spec['protocol_sha256']==sha(registration)
        enrolled = [row for row in reg['rows'] if row['seed'] == seed]
        assert len(enrolled) == 1 and seed in range(80965501,80965541)
        assigned = enrolled[0]
        assert seed==spec['controller_seed']==physical['seed']==seal['seed'] and spec['role']==physical['role']==assigned['role']=='held'
        assert spec['arm']==assigned['arm'] and assigned['arm'] in ('lambda1-parent','mixed-ppo')
        assert binding['release_sha256']==digest(read('branch-release.json')) and digest(read('vm-anchor.json'))==reg['anchor_sha256']==spec['anchor_sha256']
        assert digest(read('vm-policy-anchor.json'))==reg['policy_anchor_sha256'] and spec['policy_sha256']==physical['policy_sha256']==assigned['policy_sha256']==sha(model)
        assert spec['source_policy_sha256']==physical['source_policy_sha256']==src['policy_sha256']=='8a0e4ff3a8e6f552db9bc5d2cf34f15db9a2c12d41454e90a9e297f99099b0b0'
        assert (spec['policy_sha256']==spec['source_policy_sha256'])==(assigned['arm']=='lambda1-parent')
        assert all(binding.get(k)==v for k,v in spec.items())
        assert pre['tick']==2201 and pre['stage']==1 and pre['spares']==2 and pre['bombs']==1 and anchor['tick']==pre['tick']
        assert sha(shard)==physical['sha256']==seal['fragment_sha256']
        tr=npz(shard.read_bytes());n=len(tr['action']);assert tr['frames'].shape==(n,112,96) and tr['frames'].dtype==np.uint8
        assert np.array_equal(tr['predecessor_frame'],prefix['frames'][-1]) and digest(tr['frames'][0].tobytes())==pre['camera_sha256']
        assert np.array_equal(tr['prior_action'],prefix['action'][-1]) and np.array_equal(tr['prior_logp'],prefix['logp'][-1]) and np.array_equal(tr['prior_value'],prefix['value'][-1])
        whole=guest_npz({k:np.concatenate([prefix[k],tr[k]])for k in ('frames','action','logp','value')})
        assert digest(whole)==seal['full_behavior_sha256']==src['behavior_sha256']==physical['native_source_sha256']['behavior.npz']
        for name,h in physical['native_source_sha256'].items():
            if name!='behavior.npz':assert digest(read(name))==h,name
        assert seal['signed_clock_sha256']==digest(read('clock.tsv'))
        timing=[json.loads(line)for line in read('timing.jsonl').splitlines()]
        indices=seal['source_timing_row_indices'];plays=[timing[i]for i in indices[:-1]];last=timing[indices[-1]]
        assert len(plays)==n and all(row['phase']=='policy'for row in plays)
        normals=witness_hud(seal,read,tr,last)
        g,q,f,clock=[np.genfromtxt(io.BytesIO(read(k)),names=True,dtype=None,encoding='ascii',ndmin=1) for k in ('gates.tsv','quiet-clock.tsv','frozen.tsv','clock.tsv')]
        offset=physical['offset'];assert offset==len(prefix['action']);g,q,f=g[offset:],q[offset:],f[offset:]
        assert len(g)==len(q)==len(f)==n
        assert np.array_equal(g['camera_observed'],tr['native_tick']) and np.array_equal(g['apply_observed'],tr['native_tick']) and np.array_equal(q['tick'],tr['native_tick']) and np.array_equal(f['tick'],tr['native_tick'])
        assert np.all(q['serial_after_input']>q['serial_before_camera']) and np.all(f['stop_start_ns']<=f['stopped_ns']) and np.all(f['stopped_ns']<=f['heartbeat_ns']) and np.all(f['heartbeat_ns']<=f['camera_ns'])
        assert np.all(f['camera_ns']<=f['cmd_ns']) and np.all(f['cmd_ns']<=f['ack_ns']) and np.all(f['ack_ns']<=q['quiescent_after_input_ns']) and np.all(q['quiescent_after_input_ns']<=f['resume_ns'])
        steps=tr['native_ticks_under_command'];assert np.array_equal(steps,tr['native_next_tick']-tr['native_tick'])
        prior=np.concatenate([tr['predecessor_frame'][None],tr['frames'][:-1]])
        x=np.stack([tr['frames'],np.abs(tr['frames'].astype(np.int16)-prior).astype(np.uint8)],1);lp=[];v=[]
        with torch.inference_mode():
            for pos in range(0,n,64):
                y=actor(torch.from_numpy(x[pos:pos+64]).float()/255.);a=torch.from_numpy(tr['action'][pos:pos+64].astype(np.int64))
                lp.append(torch.stack([head.log_softmax(-1).gather(1,a[:,j:j+1]).squeeze(1)for j,head in enumerate(y[:4])],1).numpy())
                v.append(y[4].numpy().reshape(-1))
            logerr=float(np.max(np.abs(np.concatenate(lp)-tr['logp'])));verr=float(np.max(np.abs(np.concatenate(v)-tr['value'])))
            terminal=bool(tr['terminated'][-1]);assert not tr['terminated'][:-1].any()
            bf=tr['boundary_frame'];assert digest(bf.tobytes())==physical['boundary_camera_sha256']
            bound=None if terminal else float(actor(torch.from_numpy(np.stack((bf,np.abs(bf.astype(np.int16)-tr['frames'][-1]).astype(np.uint8)))[None]).float()/255.)[-1].item())
        kind=physical['outcome']
        if not terminal:
            assert last['phase']=='fragment_boundary' and last['camera_sha256']==digest(bf.tobytes()) and last['camera_native_tick']==tr['native_next_tick'][-1] and last['stage']==tr['native_next_stage'][-1]
            assert last['cutoff_reason']==kind
        observed_term=(terminal and kind=='game_over' and last['phase']=='terminal' and src['outcome']=='original_continue_prompt' and (tr['next_spares'][-1]==0))
        cutoff=bool(observed_term or (kind=='life_loss_cutoff' and np.all(tr['spares']>=pre['spares']) and tr['next_spares'][-1]<pre['spares']) or (kind=='native_tick_cutoff' and tr['native_next_tick'][-1]-pre['tick']>=3000))
        observed={}
        proven=(clock['stable']==1)&(clock['mode']==0)&(clock['flags']==4)&(clock['buffer_read']==1)&(clock['saved_current']==clock['word'])&(clock['cursor']==clock['origin']+36+2*clock['tick'])
        for r in clock[proven & (clock['tick']>=pre['tick'])]:
            observed.setdefault((int(r['stage_index'])+1,int(r['tick'])),set()).add(int(r['word']))
        saved={}
        for stage in set(tr['native_stage']):
            for ext in ['.bin','.partial']:
                name=f'native-record-stage-{stage}'+ext
                if name not in files:continue
                buf=read(name);arr=np.frombuffer(buf[4:],dtype='<u2')
                if int.from_bytes(buf[:4],'little')==len(arr):saved[int(stage)]=arr;break
        move,focus,shot,bomb=tr['action'].astype(int).T
        masks=(move%3==0)*1+(move%3==2)*2+(move//3==0)*4+(move//3==2)*8+focus*16+shot*32+bomb*64
        assert np.array_equal(g['x11_mask'],masks)
        requested=(move%3==0)*64+(move%3==2)*128+(move//3==0)*16+(move//3==2)*32+focus*4+shot+bomb*2
        prev=anchor['consumed_word']&247;stale=[];bad=[];matched=0;native_slots=0
        for i in range(n):
            stage=int(tr['native_stage'][i]);tick=int(tr['native_tick'][i]);word=int(requested[i]);actual=[]
            for j in range(int(steps[i])):
                original_tick=tick+1+j;seen=set(observed.get((stage,original_tick),()));buf=saved.get(stage,())
                if original_tick<len(buf):seen.add(int(buf[original_tick]))
                actual.append(next(iter(seen)) if len(seen)==1 else -1)
            assert actual[:4]==tr['native_word_slots'][i,:min(4,len(actual))].tolist()
            native_slots+=sum(k>=0 for k in actual);matched+=word in actual
            if len(actual)>=4 and prev!=word and all(k==prev for k in actual):stale.append(i)
            if (not terminal or i!=n-1) and actual!=[word]*4:bad.append(dict(index=i,tick=tick,prior_word=int(prev),expected=word,actual=actual))
            prev=word
        windows=n-int(terminal);fraction=(windows-len(bad))/windows if windows else 0.
        short=windows<200 and len(bad)==1 and bad[0]['index']>=60 and bad[0]['expected']!=bad[0]['prior_word'] and bad[0]['actual']==[bad[0]['prior_word']]+[bad[0]['expected']]*3
        first=all(b['index']>=min(windows,60) for b in bad)
        eligible=bool(cutoff and n>0 and first and matched==n and not stale and (fraction>=.995 or short) and native_slots==steps.sum() and logerr<=1e-5 and verr<=1e-5 and normals/n>=.97)
        control=dict(per_credit_pass=eligible,observed_first_60_exact=first,first_min_60_exact=first,first_60_exact=first,first_sixty_exact_original_windows=min(60,windows)-sum(b['index']<60 for b in bad),first_sixty_denominator=min(60,windows),
                     qualified_via_short_exception=bool(short and fraction<.995),short_partial_first_slots=bad,exact_windows=windows-len(bad),total_windows=windows,native_words_reconstructed=native_slots,all_masks_consumed=matched==n)
        canon=dict(physical,raw_receipt_sha256=digest(read('fragment-receipt.json')),qualified=eligible,qualified_via_short_exception=bool(short and fraction<.995),exact_four_tick_fraction=fraction,control_evidence=control,
                   wholly_stale_windows=len(stale),action_masks_with_native_consumption=matched,policy_sha256=assigned['policy_sha256'],source_kind='anchor',anchor_sha256=reg['anchor_sha256'],continues=0,role='held',arm=assigned['arm'],
                   game_sha256=reg['game_sha256'],outcome=kind,boundary_camera_sha256=digest(bf.tobytes()),normal_hud_rows=normals,max_actor_logp_abs=logerr,max_critic_abs=verr,behavior_boundary_value=bound,compact_archive_sha256=sha(compact),
                   source_seal_sha256=digest(read('compact-seal.json')),physical_origin_source_sha256=seal['full_behavior_sha256'])

        return canon

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration',required=True,type=Path)
    p.add_argument('--prefix',required=True,type=Path)
    p.add_argument('--model',required=True,type=Path)
    p.add_argument('--normal-glyph',required=True,type=Path)
    p.add_argument('--compact',required=True,type=Path)
    p.add_argument('--fragment',required=True,type=Path)
    p.add_argument('--seed',required=True,type=int)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    registration=a.registration;reg=json.loads(registration.read_bytes());prefix=npz(a.prefix.read_bytes())
    model=a.model;rank_template=np.load(a.normal_glyph,allow_pickle=False)
    torch.set_num_threads(1);actor=torch.jit.load(str(model),map_location='cpu').eval()
    result=accept(a.seed,a.compact,a.fragment)
    write(a.out,result)
    print(a.seed,result['qualified'],result['exact_four_tick_fraction'],result['max_actor_logp_abs'],result['behavior_boundary_value'])
