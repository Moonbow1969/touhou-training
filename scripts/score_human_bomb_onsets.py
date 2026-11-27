"""Measure whether a held replay's next four native ticks contain a genuine Bomb-key rising edge."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scripts.checkpoint_storage import digest
from scripts.train_human_camera_policy import load_replay


def onset_labels(path):
    with np.load(path,allow_pickle=False) as z:
        future=(z['input_chunk']&2)!=0;past=(z['past_input_chunk'][:,-1:]&2)!=0
        words=np.column_stack((past,future));rises=(~words[:,:-1])&words[:,1:]
        tick=z['native_frame'].astype(int)
        keep=z['stable'].astype(bool)&np.r_[False,np.diff(tick)==4]
        return rises.any(1)[keep],int(rises[keep].sum())


def score(prob,truth):
    n=len(truth);positives=int(truth.sum())
    order=np.argsort(-prob,kind='stable')
    ends=np.flatnonzero(np.r_[np.diff(prob[order])!=0, True])
    tp=np.cumsum(truth[order])[ends];precision=tp/(ends+1)
    ap=float((np.diff(np.r_[0,tp])*precision).sum()/positives) if positives else None
    # Include the entire tie at the credit's top 5% cutoff.
    cutoff=float(prob[order[max(0,int(np.ceil(.05*n))-1)]])
    selection=prob>=cutoff
    return dict(rows=n,onset_windows=positives,prevalence=positives/n,average_precision=ap,
                top_five_percent_flagged=int(selection.sum()),top_five_percent_onsets=int(truth[selection].sum()),
                top_five_percent_false_alerts=int((~truth[selection]).sum()),top_five_percent_threshold=cutoff,
                model_argmax_alerts=int((prob>=.5).sum()),model_argmax_true_onsets=int(truth[prob>=.5].sum()))


def calibration(prob,truth):
    p=np.clip(prob.astype(np.float64),1e-12,1-1e-12)
    truth=np.asarray(truth,dtype=bool)
    return dict(rows=len(truth),positive_windows=int(truth.sum()),expected_positive_windows=float(p.sum()),
                expected_on_positives=float(p[truth].sum()),binary_nll=float(-np.where(truth,np.log(p),np.log1p(-p)).mean()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',type=Path,required=True);p.add_argument('--held',nargs='+',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2);data=[]
    for path in a.held:
        row=load_replay(path)
        target,edges=onset_labels(path)
        if len(target)!=len(row['raw']):raise ValueError('Causal human action and onset windows differ')
        data.append((row,target,edges))
    report={}
    with torch.inference_mode():
        for folder in a.models:
            meta=json.loads((folder/'report.json').read_bytes());h=digest(folder/'policy.ts')
            if h!=meta['torchscript_sha256']:raise ValueError('Changed supervised policy')
            train={x['group'] for x in meta['train']};model=torch.jit.load(str(folder/'policy.ts'),map_location=device).eval()
            report[folder.name]={}
            for row,truth,edges in data:
                if row['group'] in train:raise ValueError('Training player appeared in evaluation')
                prob=torch.cat([model(x.to(device).float()/255)[3].softmax(1)[:,1].cpu()
                                for x in torch.from_numpy(row['raw']).split(64)]).numpy()
                report[folder.name][row['name']]=dict(group=row['group'],camera_sha256=row['sha256'],model_sha256=h,
                    true_key_edges=edges,aligned=score(prob,truth),half_credit_shift=score(np.roll(prob,len(prob)//2),truth),
                    training_target_any_x=calibration(prob,row['action'][:,3]),rising_edge_target=calibration(prob,truth))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
