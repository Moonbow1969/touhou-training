"""Fit all four human camera actions, traversing every declared training window once per epoch."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from scripts.checkpoint_storage import digest
from scripts.train_human_camera_policy import evaluation, load_replay
from scripts.train_replay_camera_encoder import CameraPolicy


def save_atomic(path, value):
    tmp = path.with_suffix('.partial')
    torch.save(value, tmp)
    os.replace(tmp, path)


def write_json(path, data):
    tmp = path.with_suffix('.part')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf8')
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    bundle, out = args.bundle.resolve(), args.out.resolve()
    out.mkdir(exist_ok=True, parents=True)
    protocol = bundle/'protocol.json'
    plan = json.loads(protocol.read_bytes())
    binding = digest(protocol)
    torch.set_num_threads(2)
    torch.manual_seed(plan['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    model = CameraPolicy().to(device)
    source = bundle/'encoder.pt'
    if digest(source) != plan['encoder_sha256']:
        raise ValueError('Unexpected initial human visual encoder')
    model.load_state_dict(torch.load(source, map_location='cpu', weights_only=True))
    opt = torch.optim.AdamW([{'params':model.spatial.parameters(), 'lr':.00015},
                             {'params':model.policy.parameters(), 'lr':.0005}], weight_decay=.001)
    n = plan['decisions']
    images = torch.empty((n, 2, 112, 96), dtype=torch.uint8, device=device)
    labels = torch.empty((n, 4), dtype=torch.long, device=device)
    weights = torch.empty(n, dtype=torch.float32, device=device)
    train_rows = []
    strata = Counter()
    offset = 0
    train_players = set()
    def load(source):
        path = bundle/source['file']
        if digest(path) != source['sha256'] or digest(path.with_suffix('.json')) != source['metadata_sha256']:
            raise ValueError('Changed replay or metadata')
        row = load_replay(path)
        if (row['title'], row['group'], row['difficulty']) != (source['title'],source['group'],source['rank']):
            raise ValueError('Changed replay attribution')
        return row
    for src in plan['train']:
        row = load(src)
        end = offset + len(row['action'])
        images[offset:end].copy_(torch.from_numpy(row['raw']))
        labels[offset:end].copy_(torch.from_numpy(row['action']))
        train_rows.append(dict(name=src['name'], title=row['title'], group=row['group'], sha256=row['sha256'], decisions=end-offset))
        group = row['title'],row['difficulty']
        strata[group] += end - offset
        train_players.add(row['group'])
        offset = end
        del row
    if offset != n or len(strata) != 8 or len(set(v['sha256'] for v in train_rows)) != len(train_rows):
        raise ValueError('Incomplete unique training bank')
    begin = 0
    for row,src in zip(train_rows,plan['train']):
        end = begin + row['decisions']
        weights[begin:end] = n / (len(strata) * strata[src['title'],src['rank']])
        begin = end
    if abs(float(weights.mean())-1) > 1e-5:
        raise ValueError('Unnormalized per-title/rank weights')
    if train_players & {s['group'] for s in plan['development']}:
        raise ValueError('Replay player leakage')
    state = out/'resume.pt'
    start = 1
    history = []
    if state.exists():
        checkpoint = torch.load(state, weights_only=True, map_location=device)
        if checkpoint['protocol'] != binding:raise ValueError('Resume registration mismatch')
        model.load_state_dict(checkpoint['model']);opt.load_state_dict(checkpoint['optimizer'])
        start = checkpoint['epoch'] + 1;history = checkpoint['history']
    print('Verified', n, 'distinct real windows', strata, torch.cuda.get_device_name(), flush=True)
    batch_size = plan['batch_size']
    for epoch in range(start, plan['epochs']+1):
        ts = time.monotonic()
        torch.manual_seed(plan['seed']+epoch)
        order = torch.randperm(n, device=device)
        if not torch.equal(order.sort().values, torch.arange(n,device=device)):
            raise ValueError('Invalid epoch permutation')
        model.train()
        loss_sum = 0.
        for index, ids in enumerate(order.split(batch_size)):
            px = images[ids].float() / 255
            y = labels[ids].clone()
            flip = torch.rand(len(ids),device=device) < .5
            px[flip] = px[flip].flip(3)
            direction = y[flip,0]
            y[flip,0] = 3*(direction//3) + 2 - direction%3
            px *= .85 + .3*torch.rand(len(ids),1,1,1,device=device)
            heads = model(px)
            row_loss = sum(F.cross_entropy(h,y[:,i],label_smoothing=.025,reduction='none') for i,h in enumerate(heads))
            loss = (row_loss * weights[ids]).mean()
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            loss_sum += float(loss.detach()) * len(ids)
            if index % 200 == 0:
                print(epoch,index,min(n,(index+1)*batch_size),'/',n,float(loss.detach()),flush=True)
        history.append(dict(epoch=epoch,windows=n,total_presentations=n*epoch,mean_weighted_joint_loss=loss_sum/n,
                            wall_seconds=time.monotonic()-ts))
        if epoch in plan['milestones']:
            dest = out/f'epoch-{epoch:02}'
            dest.mkdir(exist_ok=True)
            model.eval()
            probes = {}
            with torch.no_grad():
                for src in plan['development']:
                    row = load(src)
                    probs=[np.concatenate(p) for p in zip(*[[head.softmax(1).cpu().numpy() for head in model(v.to(device).float()/255)]
                          for v in torch.from_numpy(row['raw']).split(96)])]
                    probes[src['sha256']]=dict(group=row['group'],title=row['title'],rank=row['difficulty'],metrics=evaluation(row,probs))
            cpu = CameraPolicy().eval()
            cpu.load_state_dict(model.state_dict())
            save_atomic(dest/'camera-policy.pt',cpu.state_dict())
            test=images[:4].cpu().float()/255
            with torch.no_grad():
                exported=torch.jit.trace(cpu,test[:1]);torch.testing.assert_close(exported(test),cpu(test))
            exported.save(str(dest/'staged-policy.ts'));os.replace(dest/'staged-policy.ts',dest/'policy.ts')
            write_json(dest/'report.json',dict(protocol=binding,epoch=epoch,train=train_rows,development=probes,
                sft=digest(dest/'camera-policy.pt'),torchscript_sha256=digest(dest/'policy.ts'),encoder=digest(source),history=history))
        save_atomic(state,dict(protocol=binding,epoch=epoch,model=model.state_dict(),optimizer=opt.state_dict(),history=history))
        write_json(out/'history.json',history)
        print('Completed epoch',epoch,history[-1],flush=True)


if __name__=='__main__':
    main()
