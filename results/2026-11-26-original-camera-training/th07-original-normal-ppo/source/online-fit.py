"""One real full-policy PPO increment from fresh original TH07 credits."""
from copy import deepcopy
import csv
from hashlib import sha256
import json
from pathlib import Path
import time
import struct
import numpy as np
import torch
from inspect_rpy import decompress

base=Path('/var/tmp/moonbow-th07-trial-261127')
root=base/'online'
def sha(p):return sha256(p.read_bytes()).hexdigest()
def save(p,doc):p.write_text(json.dumps(doc,indent=2)+'\n')
plan=json.loads((root/'protocol.json').read_text());cfg=plan['optimizer']
torch.manual_seed(cfg['optimizer_seed']);torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
device='cuda' if torch.cuda.is_available() else 'cpu'
parent=base/'policy/policy.ts'
assert sha(parent)==plan['parent_policy_sha256']
model=torch.jit.load(str(parent),map_location=device).eval()
def selected(heads,actions):
    return torch.stack([y.log_softmax(-1).gather(1,actions[:,h:h+1])[:,0] for h,y in enumerate(heads[:4])],1)
def replay(file):
    b=bytearray(file.read_bytes());assert b[:4]==b'T7RP'
    b[16:]=bytes((v-b[13]-7*i)%256 for i,v in enumerate(b[16:]))
    comp,exp,*slots=struct.unpack_from('<9I',b,20)
    assert comp==len(b)-84 and 0<exp<2_000_000
    p=decompress(bytes(b[84:]),exp);assert list(p[2:4])==[0,1]
    return 10*struct.unpack_from('<I',p,24)[0],[i+1 for i,n in enumerate(slots) if n]
def pack(folder,role,arm):
    r=json.loads((folder/'receipt.pending-visual.json').read_text())
    score,stages=replay(folder/'official-policy.rpy')
    with np.load(folder/'policy-trace.npz') as z:
        frame=z['frames'];act=z['action'];lp=z['logp'];value=z['value'];clock=z['clock_s']
    rows=list(csv.DictReader((folder/'controls.csv').open()))
    assert len(rows)==len(frame)==r['actions'] and rows[0]['spares']=='2' and rows[0]['bombs']=='3'
    assert (folder/'settled-continue.png').is_file() and (folder/'settled-scorecard.png').is_file()
    assert np.array_equal(act,np.array([[int(row[k]) for k in ('move','focus','shot','bomb')]for row in rows]))
    img=np.stack([frame,np.abs(frame.astype(np.int16)-np.concatenate([frame[:1],frame[:-1]]).astype(np.int16)).astype(np.uint8)],1)
    lives=[int(row['spares']) for row in rows]
    death=[];spares=2
    for i in range(1,len(lives)-2):
        if lives[i:i+3]==[spares-1]*3 and spares:
            death.append(max(0,i-1));spares-=1
    if spares:raise ValueError(f'Unresolved physical lives in {folder}')
    death.append(len(frame)-1)
    reward=np.full(len(frame),.002,np.float32);reward[death]-=1
    done=np.zeros(len(frame),bool);done[-1]=True
    np.savez_compressed(folder/'ppo-reward.npz',reward=reward,terminated=done,life_loss_action=np.array(death))
    receipt=dict(**r,role=role,arm=arm,host='momentum',rank='Normal',shot='Reimu A',
        original_replay_score=score,stage_checkpoints=stages,stage2_reached=2 in stages,
        native_opening=[2,3],last_hud_spares=lives[-1],last_hud_bombs=int(rows[-1]['bombs']),
        last_visible_continue='Original prompt; controller released all held keys',evidence_sha256={k:sha(folder/k) for k in [
        'controls.csv','policy-trace.npz','official-policy.rpy','full-stock-opening.png','settled-continue.png','settled-scorecard.png','ppo-reward.npz']})
    save(folder/'receipt.json',receipt)
    delta=0.;next_value=0.;adv=np.empty(len(frame),np.float32)
    for t in range(len(frame)-1,-1,-1):
        delta=reward[t]+cfg['gamma']*next_value-value[t]+cfg['gamma']*cfg['gae_lambda']*delta
        adv[t]=delta;next_value=value[t]
    return (img,act,lp,value,adv,adv+value),receipt

def fit():
    folders=[root/f'train-{s}' for s in plan['train']]
    deadline=time.monotonic()+820
    while not all((f/'receipt.pending-visual.json').exists() for f in folders):
        if time.monotonic()>deadline:raise TimeoutError('No complete four-credit training batch')
        time.sleep(.5)
    packed=[pack(f,'train',plan['parent_arm']) for f in folders]
    rc=[r for _,r in packed]
    assert all(r['policy_sha256']==sha(parent) and r['role']=='train' for r in rc)
    x,a,old_logp,old_value,adv,ret=[torch.from_numpy(np.concatenate(v)).to(device) for v in zip(*(d for d,_ in packed))]
    old_logp=old_logp.float();ret=ret.float();old_value=old_value.float()
    def evaluate():
        with torch.no_grad():
            result=[];vals=[]
            for i in range(0,len(x),128):
                h=model(x[i:i+128].float()/255);result.append(selected(h,a[i:i+128]));vals.append(h[4].flatten())
            return torch.cat(result),torch.cat(vals)
    before,v=evaluate()
    pe=float((before-old_logp).abs().max());ve=float((v-old_value).abs().max())
    if max(pe,ve)>4e-5:raise ValueError(f'Original actor mismatch {pe} {ve}')
    adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    optim=torch.optim.Adam(model.parameters(),lr=cfg['adam_lr'])
    old=old_logp.sum(1);steps=0;log=[];halt=False
    for epoch in range(cfg['epochs']):
        order=torch.randperm(len(x),device=device)
        prev=deepcopy(model.state_dict());steps_before=steps
        for k,ids in enumerate(order.split(cfg['minibatch'])):
            h=model(x[ids].float()/255);new=selected(h,a[ids]).sum(1);ratio=(new-old[ids]).exp()
            pol=-torch.minimum(ratio*adv[ids],ratio.clamp(1-cfg['clip'],1+cfg['clip'])*adv[ids]).mean()
            value=(h[4].flatten()-ret[ids]).square().mean()
            entropy=-sum((z.softmax(1)*z.log_softmax(1)).sum(1).mean()for z in h[:4])
            loss=pol+cfg['value_coef']*value-cfg['entropy']*entropy
            optim.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),.5);optim.step();steps+=1
            if (k+1)%4==0 or (k+1)*cfg['minibatch']>=len(x):
                now,_=evaluate();d=now.sum(1)-old;kl=float((d.exp()-1-d).mean())
                ok=np.isfinite(kl) and kl<=cfg['kl_stop']
                log.append(dict(epoch=epoch+1,step=steps,approx_joint_kl=kl,accepted=bool(ok),loss=float(loss.detach())))
                if not ok:
                    model.load_state_dict(prev);steps=steps_before;halt=True;break
                prev=deepcopy(model.state_dict());steps_before=steps
        if halt:break
    if not steps:raise RuntimeError('No update passed registered KL guard')
    path=root/'policy';path.mkdir(exist_ok=True)
    model.cpu().save(str(path/'policy.ts'));torch.save(model.state_dict(),path/'state_dict.pt')
    childsha=sha(path/'policy.ts');assert childsha!=sha(parent)
    report=dict(algorithm='Joint four-head PPO',policy_sha256=childsha,parent_sha256=sha(parent),physical_batch=len(rc),
        train_decisions=len(x),epochs_budget=cfg['epochs'],accepted_optimizer_steps=steps,max_initial_logp_error=pe,
        max_initial_critic_error=ve,configuration=cfg,protocol_sha256=sha(root/'protocol.json'),credits=rc,history=log)
    save(path/'report.tmp',report);(path/'report.tmp').replace(path/'report.json')
    print('FROZEN',childsha,'steps',steps,'original decisions',len(x),flush=True)

if __name__=='__main__':fit()
