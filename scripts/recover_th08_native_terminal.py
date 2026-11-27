"""Recover the first verified TH08 continue frame from the uncut physical observations."""
import cv2
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze


def find_terminal(folder):
    receipt=json.loads((folder/'receipt.json').read_bytes())
    if digest(folder/'native-observers.npz')!=receipt['native_observers_sha256']:
        raise ValueError('Original observer bytes changed')
    rows=[json.loads(x) for x in (folder/'timing.jsonl').read_bytes().splitlines()]
    tool=Path(__file__).resolve().parents[1]/'tools/rollout-worker-probe/th08-trial'
    with np.load(tool/'continue-edges.npz') as ref: edge=ref['edge'];mask=ref['mask']
    text=np.load(tool/'continue-text-mask.npy',allow_pickle=False)
    with np.load(folder/'native-observers.npz',allow_pickle=False) as src:
        prompt=src['prompt']
    # Filter cheap recorded statistics, then independently check both structures on the lossless camera crop.
    had_zero=False
    for j,row in enumerate(rows):
        if row['phase']=='transition':
            continue
        if row.get('hud',0)>=.95 and row.get('spares')==0:
            had_zero=True
        if not had_zero or row.get('spares')!=0 or row.get('continue_text',0)<.85 or row.get('continue_structure',0)<.78:
            continue
        original=prompt[j]
        bright=float((original[4:36,4:216][text]>150).mean())
        box=original[3:37,2:215].astype(np.float32);e=(box-cv2.GaussianBlur(box,(0,0),1.35))[mask]
        r=edge[mask];corr=float(e@r/(np.linalg.norm(e)*np.linalg.norm(r)+1e-8))
        if bright>=.85 and corr>=.78:
            return j,dict(source_prompt_bright=bright,source_prompt_structural_correlation=corr)
    return None,dict(reason='No qualifying continue prompt after verified zero spare lives')


def recover(folder,dest):
    receipt=json.loads((folder/'receipt.json').read_bytes())
    index,evidence=find_terminal(folder)
    if index is None:
        return None,evidence
    rows=[json.loads(x) for x in (folder/'timing.jsonl').read_bytes().splitlines()]
    if index==len(rows)-1 and rows[index]['phase']=='terminal' and receipt['outcome']=='original_continue_prompt':
        return folder,evidence
    if digest(folder/'behavior.npz')!=receipt['behavior_sha256'] or digest(folder/'timing.jsonl')!=receipt['timing_sha256']:
        raise ValueError('Original physical actions or chronology changed')
    n=sum(r['phase']=='policy' for r in rows[:index])
    if n<3:
        raise ValueError('Missing physical input before original continue')
    dest.mkdir(parents=True,exist_ok=True)
    for name in ('protocol.json','chosen-rank.png','first-policy-frame.png'):
        if not (dest/name).exists():os.link(folder/name,dest/name)
    valid=rows[:index]+[{k:v for k,v in rows[index].items() if k not in ('mask','x_applied_ns')}]
    valid[-1]['phase']='terminal'
    (dest/'timing.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in valid),encoding='utf8')
    with np.load(folder/'behavior.npz',allow_pickle=False) as src:
        arrays={k:src[k][:n] for k in ('frames','action','logp','value')}
    np.savez_compressed(dest/'behavior.npz',**arrays)
    with np.load(folder/'native-observers.npz',allow_pickle=False) as src:
        np.savez_compressed(dest/'native-observers.npz',hud=src['hud'][:index+1],prompt=src['prompt'][:index+1])
    cap=cv2.VideoCapture(str(folder/'original-observations.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,bgr=cap.read();cap.release()
    if not ok:raise ValueError('No physical video frame at recovered first terminal')
    Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).save(dest/'terminal.png')
    updated={**receipt,'outcome':'original_continue_prompt','physical_decisions':n,
             'uncut_physical_policy_decisions':receipt['physical_decisions'],
             'uncut_physical_source_receipt_sha256':digest(folder/'receipt.json'),
             'recovered_native_first_terminal_frame':index,'native_first_terminal_evidence':evidence,
             'displayed_terminal_source':'Decoded original camera movie frame. Numerical keyboard, stock and text verification uses lossless source and observer arrays.',
             'observed_duration_s':(valid[-1]['monotonic_ns']-valid[0]['monotonic_ns'])/1e9,
             'direction_histogram':np.bincount(arrays['action'][:,0],minlength=9).tolist()}
    for file,key in [('behavior.npz','behavior_sha256'),('native-observers.npz','native_observers_sha256'),
                     ('timing.jsonl','timing_sha256'),('terminal.png','terminal_sha256')]:
        updated[key]=digest(dest/file)
    freeze(dest/'receipt.json',updated)
    return dest,evidence
