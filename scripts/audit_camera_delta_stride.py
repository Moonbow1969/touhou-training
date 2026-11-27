"""Compare four- and eight-tick image differences at the same human training frames."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.collect_th10_full_action import save


@torch.inference_mode()
def audit(args):
    torch.set_num_threads(2)
    model=torch.jit.load(str(args.policy),map_location='cpu').eval()
    train=json.loads(args.training_report.read_bytes())['train']
    details=[]
    for source in train:
        file=args.replays/source['name']
        assert hashlib.sha256(file.read_bytes()).hexdigest()==source['sha256']
        with np.load(file) as z:
            frame=z['native_frame'].reshape(-1).astype(np.int64);x=z['obs'];truth=z['action'][:,0].astype(np.int64)
            valid=z['stable'].astype(bool).reshape(-1) if 'stable' in z.files else np.ones(len(x),dtype=bool)
        ids=np.nonzero(valid[2:]&valid[1:-1]&valid[:-2]&(np.diff(frame)[1:]==4)&(np.diff(frame)[:-1]==4))[0]+2
        assert len(ids)>0
        batches={k:[] for k in (1,2)};means={1:[],2:[]}
        for start in range(0,len(ids),32):
            i=ids[start:start+32];cur=torch.from_numpy(x[i]).float()/255
            for lag in (1,2):
                change=(cur-torch.from_numpy(x[i-lag]).float()/255).abs()
                logits=model(torch.stack((cur,change),1))[0]
                batches[lag].append(logits.log_softmax(1));means[lag].extend(change.mean((1,2)).tolist())
        row=dict(replay=file.name,player=source['group'],n=len(ids),changed_direction_targets=int((truth[ids]!=truth[ids-1]).sum()))
        modes={}
        for lag in (1,2):
            logp=torch.cat(batches[lag]);direction=logp.argmax(1).numpy();modes[lag]=direction
            correct=direction==truth[ids]
            changed=truth[ids]!=truth[ids-1]
            row[str(lag*4)+'_ticks']=dict(movement_correct=int(correct.sum()),movement_accuracy=float(correct.mean()),
                changed_correct=int(correct[changed].sum()),changed_accuracy=float(correct[changed].mean()),
                mean_movement_nll=-float(logp[torch.arange(len(ids)),torch.from_numpy(truth[ids])].mean()),
                mean_difference_0_to_1=float(np.mean(means[lag])))
        row['modal_action_switch_when_changing_only_lag']=float((modes[1]!=modes[2]).mean());details.append(row)
    result=dict(policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),replays=details,
        scope='Only human TRAIN players listed in the frozen original camera SFT report. Compare the same observations, exact following human movement, current pixels, and sample index. Replace the reference image for the absolute difference with four or eight native simulation ticks ago. This measures offline input sensitivity, not original-game survival or the native frame speed of any Linux worker.')
    save(args.out,result);print(json.dumps(details,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('policy','training-report','replays','out'):parser.add_argument('--'+key,type=Path,required=True)
    audit(parser.parse_args())
