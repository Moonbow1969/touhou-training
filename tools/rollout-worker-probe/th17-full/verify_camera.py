"""Independently replay an entire original TH17 camera trajectory against its frozen actor."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import torch

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def audit(episode, model_path):
    receipt=json.loads((episode/'receipt.json').read_bytes())
    if receipt['trajectory_sha256']!=sha(episode/'behavior.npz') or receipt['policy_sha256']!=sha(model_path):
        raise ValueError('Declared trajectory or checkpoint hash differs')
    model=torch.jit.load(str(model_path),map_location='cpu').eval()
    history_input=len(model.forward.schema.arguments)==3
    torch.set_num_threads(2)
    with np.load(episode/'behavior.npz') as z, torch.inference_mode():
        camera=np.ascontiguousarray(z['frames'])
        if hashlib.sha256(camera.tobytes()).hexdigest()!=receipt['frames_sha256']:
            raise ValueError('Pre-action raw camera digest differs')
        a=torch.from_numpy(z['action']).long();frame=torch.from_numpy(camera).float()/255
        change=torch.zeros_like(frame);change[1:]=(frame[1:]-frame[:-1]).abs();features=torch.stack((frame,change),dim=1)
        if history_input!=('previous_move' in z):
            raise ValueError('The checkpoint input signature and native H1 trace differ')
        prior=None
        if history_input:
            from direction_history import previous_move_fields
            trace=episode/'decisions.tsv'
            if sha(trace)!=receipt['decisions_sha256']:
                raise ValueError('Changed original H1 trace')
            raw=np.genfromtxt(trace,names=True,delimiter='\t',dtype=None,encoding='utf8',ndmin=1)
            fields=previous_move_fields(raw,a.numpy())
            if 'previous_move' not in fields or not np.array_equal(fields['previous_move'],z['previous_move']):
                raise ValueError('Missing or inconsistent H1 wire-byte sequence')
            prior=torch.from_numpy(fields['previous_move']).long()
        logged=z['logp'];value=z['value'];errors=[];critic=[]
        for i in range(0,len(frame),16):
            heads=model(features[i:i+16],prior[i:i+16]) if history_input else model(features[i:i+16])
            selected=a[i:i+16]
            p=torch.stack([torch.log_softmax(head,dim=1).gather(1,selected[:,j:j+1]).flatten() for j,head in enumerate(heads[:4])],dim=1)
            v=heads[4].flatten().numpy() if len(heads)==5 else np.zeros(len(p))
            errors.append(abs(p.numpy()-logged[i:i+16]).max(axis=0));critic.append(abs(v-value[i:i+16]).max())
        head_error=np.max(errors,axis=0)
        if max(head_error.max(),max(critic))>=4e-4:raise ValueError('Frozen actor mismatches original sample')
    result=dict(actions=len(camera),policy_sha256=sha(model_path),trajectory_sha256=receipt['trajectory_sha256'],mean_direction_non_neutral=float((a[:,0]!=4).float().mean()),
                model_input='actual pre-action grayscale plus consecutive absolute difference, initial difference zero',maximum_head_logp_error=head_error.tolist(),maximum_value_error=float(max(critic)),torch=torch.__version__)
    if history_input:
        result['verified_previous_move_rows']=len(prior)
        result['first_previous_move']=int(prior[0])
    (episode/'independent-policy-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('episode',type=Path);p.add_argument('checkpoint',type=Path);args=p.parse_args()
    print(json.dumps(audit(args.episode,args.checkpoint),indent=2))
