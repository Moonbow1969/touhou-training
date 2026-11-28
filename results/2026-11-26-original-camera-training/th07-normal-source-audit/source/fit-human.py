"""Make one exact-build TH07 Normal Stage 1 camera pilot from two disjoint replay sources."""
import hashlib,json,struct,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
base=Path('/var/tmp/moonbow-th07-trial-261127')
root=base/'human'
sys.path.insert(0,str(base/'online'))
from inspect_rpy import decompress

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(path,doc):path.write_text(json.dumps(doc,indent=2)+'\n')
def decode(path):
    data=bytearray(path.read_bytes())
    assert data[:6].hex()=='543752501110'
    data[16:]=bytes((v-data[13]-7*i)%256 for i,v in enumerate(data[16:]))
    size,n,*offsets=struct.unpack_from('<9I',data,20);assert size==len(data)-84
    plain=decompress(bytes(data[84:]),n)
    assert list(plain[2:4])==[0,1]
    beg,end=offsets[0]-84+40,offsets[1]-84
    return np.frombuffer(plain[beg:end],'<u2').reshape(-1,2)[:,0].copy()
def action_at(source,indices):
    keys=source[indices]
    def bit(mask):return (keys&mask!=0).astype(np.int64)
    movement=4+3*(bit(32)-bit(16))+bit(128)-bit(64)
    rising=np.concatenate([np.array([False]),((source[1:]&2)!=0)&((source[:-1]&2)==0)])
    bomb=np.stack([rising[indices+j] for j in range(4)]).any(0).astype(np.int64)
    return np.stack([movement,bit(4),bit(1),bomb],1)
def pack(role):
    p=root/('capture-'+role)
    with np.errstate(over='ignore'): t=np.loadtxt(p/'final-clock.csv',delimiter=',')[:,0].astype(np.int64)
    state=np.fromfile(p/'final-words.bin','<u4').reshape(-1,6)
    imgs=np.memmap(p/'final-pix.bin',dtype='u1',mode='r').reshape(-1,112,96)
    source=decode(p/'th7_01.rpy');assert len(imgs)==len(t)==len(state) and np.array_equal(t,state[:,0])
    assert ((t[1:]-t[:-1])>0).all()
    ok=(t>0)&(t<len(source)-4)
    current=state[:,1]&247;previous=state[:,2]&247
    times=t[ok];observed=current[ok];act=source[times]&247
    rate=(observed==act).mean()
    shadow=state[:,3].astype('int64')
    print(role,len(imgs),'tick',t[0],t[-1],'n source',len(source),'current match',rate,'shadow match',np.mean(shadow[ok]==times),flush=True)
    assert rate>.985 and len(times)>len(source)//5
    idx=np.arange(1,len(t))
    idx=idx[(t[idx]>=8)&(t[idx]+4<len(source))&(t[idx]-t[idx-1]==4)]
    good=(current[idx]==(source[t[idx]]&247))&(previous[idx]==(source[t[idx]-1]&247))&(current[idx-1]==(source[t[idx-1]]&247))
    idx=idx[good]
    X=np.stack([imgs[idx],np.abs(imgs[idx].astype('i2')-imgs[idx-1].astype('i2')).astype('u1')],1)
    y=action_at(source,t[idx]+1)
    prior=action_at(source,t[idx]-3)
    y_native=source[t[idx]+1]
    out=p/'paired.npz'
    np.savez_compressed(out,native_tick=t[idx],source_replay_sha256=digest(p/'th7_01.rpy'),observation=X,action=y,previous_expert_action=prior,
        actual_native_keys=current[idx].astype('u2'),future_native_keys=y_native,previous_native_keys=previous[idx].astype('u2'))
    info=dict(role=role,observer_clock='0x11c9ae8',original_active_key='0x11c9d24',
        original_previous_key='0x11c9d2c',native_clock_image_above_below_identical=True,raw_observations=len(t),first_clock=int(t[0]),last_clock=int(t[-1]),
        replay_ticks=len(source),n_native_checked=int(ok.sum()),same_tick_key_matches=int((observed==act).sum()),same_tick_key_fraction=float(rate),
        previous_tick_key_fraction=float((previous[ok]==(source[times-1]&247)).mean()),native_timers_agree=float((shadow[ok]==times).mean()),
        four_tick_pairs=len(idx),labeled_tick_range=[int(t[idx[0]]),int(t[idx[-1]])],switches=int(np.sum(y[:,0]!=prior[:,0])),
        focused=int(y[:,1].sum()),shots=int(y[:,2].sum()),bomb_press_events=int(y[:,3].sum()),
        opposite_inputs=int(np.sum(((y_native&48)==48)|((y_native&192)==192))),source_sha256=digest(p/'th7_01.rpy'),
        compressed_sha256=digest(out),raw_sha256={k:digest(p/k) for k in ('final-clock.csv','final-words.bin','final-pix.bin','all.tsv','matched-stage-selector.png','matched-opening.png')})
    return torch.from_numpy(X),torch.from_numpy(y),torch.from_numpy(prior),info
def metrics(model,X,y,prior,dev):
    vals=[]
    with torch.no_grad():
        for batch in X.split(96):
            vals.append([v.cpu() for v in model(batch.to(dev).float()/255)[:4]])
    heads=[torch.cat([r[j] for r in vals])for j in range(4)]
    pred=torch.stack([v.argmax(1)for v in heads],1)
    match=(pred==y).float()
    losses=torch.stack([F.cross_entropy(v,y[:,j],reduction='none') for j,v in enumerate(heads)],1)
    changed=y[:,0]!=prior[:,0]
    onset=(prior[:,0]==4)&(y[:,0]!=4)
    return dict(n=len(y),n_change=int(changed.sum()),n_onset=int(onset.sum()),
                nll_by_head=losses.mean(0).tolist(),accuracy_by_head=match.mean(0).tolist(),four_head_joint_nll=float(losses.sum(1).mean()),
                movement_change_acc=float(match[changed,0].mean()),movement_change_nll=float(losses[changed,0].mean()),onset_acc=float(match[onset,0].mean()),
                previous_direction_baseline_acc=float((prior[:,0]==y[:,0]).float().mean()),expert_movement_counts=torch.bincount(y[:,0],minlength=9).tolist(),
                predicted_movement_counts=torch.bincount(pred[:,0],minlength=9).tolist(),expert_bomb_positive=int(y[:,3].sum()),
                mean_bomb_probability=float(heads[3].softmax(1)[:,1].mean()))
def main():
    for i in range(300):
        if all((root/('capture-'+r)/'final.done').is_file()for r in ['train','held']):break
        time.sleep(1)
    else:raise RuntimeError('Both native Stage1 archives required')
    tr=pack('train');he=pack('held')
    parent=base/'policy/policy.ts';assert digest(parent)=='a6e1dec1625af50e214f141a4e405795fbdae905d78c4b887e423de0bca6a77e'
    assert tr[-1]['source_sha256']=='219b2a89be16720447df65fd035618f5926af53d7e8337d5f61e07842ba3fb94'
    assert he[-1]['source_sha256']=='9da50f8af1d33d3db5cfdb2f80c9d16143d9538ec0be3ceab33f223c31a5a64f'
    torch.set_num_threads(1);torch.manual_seed(2070772);dev='cuda' if torch.cuda.is_available() else 'cpu'
    model=torch.jit.load(str(parent),map_location=dev).eval()
    before={k:metrics(model,*batch[:3],dev)for k,batch in [('train',tr),('held',he)]}
    optim=torch.optim.Adam(model.parameters(),lr=1e-5)
    X,y=tr[:2];history=[]
    for epoch in range(2):
        for ids in torch.randperm(len(y)).split(64):
            heads=model(X[ids].to(dev).float()/255);targets=y[ids].to(dev)
            loss=sum(F.cross_entropy(h,targets[:,i])for i,h in enumerate(heads[:4]))
            optim.zero_grad();loss.backward();optim.step();history.append(float(loss.detach()))
    after={k:metrics(model,*batch[:3],dev)for k,batch in [('train',tr),('held',he)]}
    out=root/'sft';out.mkdir(exist_ok=True);model.cpu().save(str(out/'policy.ts'))
    write(out/'report.json',dict(game='Original TH07 0.11 Normal Reimu A Stage 1',source_roles={'train':'Boredperson','held':'zylo'},
        aim='Small two-credited-replay pilot, one training source/one held source. No native gameplay improvement inferred from held action prediction.',
        training=dict(adam_lr=1e-5,epochs=2,minibatch=64,seed=2070772,steps=len(history),loss='Sum of four mean CE losses on uniform complete pass',
            input='96x112 gray image and unsigned absolute difference from exact t-4; labels source direction/focus/shoot at t+1 and Bomb rising edge t+1..t+4',
            objective_by_minibatch=history),native=[tr[-1],he[-1]],parent_sha256=digest(parent),policy_sha256=digest(out/'policy.ts'),held_before=before['held'],held_after=after['held'],
        train_before=before['train'],train_after=after['train'],device=dev))
    print('SEALED',len(history),before['held']['nll_by_head'],after['held']['nll_by_head'],flush=True)
if __name__=='__main__':main()
