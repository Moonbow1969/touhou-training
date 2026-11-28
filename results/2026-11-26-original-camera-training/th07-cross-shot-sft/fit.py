"""Matched TH07 data-mixture ablation with original replay sources and a pixel-only deployment."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def directions(words):
    up, down, left, right = [(words & bit != 0).astype(np.int64) for bit in (16, 32, 64, 128)]
    return 4 + 3 * (down - up) + right - left


def load_source(root, meta):
    p = root / meta['path']
    assert sha(p/'source-audit.json') == meta['audit_sha256']
    audit = json.loads((p/'source-audit.json').read_bytes())
    assert audit['original_replay_sha256'] == meta['source_sha256']
    for name in ('positions.npz','gray.raw.gz'):
        assert sha(p/name) == audit['artifacts_sha256'][name]
    with np.load(p/'positions.npz', allow_pickle=False) as z:
        t = z['native_tick']
        j = z['causal_pair_rows']
        xy = z['xy_native'][j]
        a = z['action_next_fourheads'].astype('i8')
        words = z['stage_replay_words'] if 'stage_replay_words' in z else z['stage1_replay_words']
        assert np.array_equal(z['native_word'], words[t]) and np.array_equal(z['previous_native_word'], words[t-1])
        if 'xy_qualified' in z: assert np.all(z['xy_qualified'][j] & z['xy_qualified'][j-1])
    with gzip.open(p/'gray.raw.gz','rb') as f:
        gray = np.frombuffer(f.read(), 'u1').reshape(-1,112,96)
    assert len(gray) == len(t) and np.all(t[j] - t[j-1] == 4)
    now = t[j]
    assert np.array_equal(a[:,0], directions(words[now+1]))
    delta = np.abs(gray[j].astype('i2') - gray[j-1]).astype('u1')
    x = np.stack((gray[j],delta),1)
    change = a[:,0] != directions(words[now-3])
    return tuple(torch.from_numpy(v) for v in (x, a, xy, change, now))


class ShotPolicy(nn.Module):
    def __init__(self,parent):
        super().__init__()
        self.spatial = parent.camera.spatial
        self.path = nn.ModuleList(list(parent.camera.policy.children()))
        self.player = parent.camera.player
        self.value = parent.value
        self.shot = nn.Embedding(6,24)
        nn.init.zeros_(self.shot.weight)

    def outputs(self,images,shot_ids):
        spatial = self.spatial(images) + self.shot(shot_ids)[:,:,None,None]
        h = self.path[1](self.path[0](spatial))
        # An exact partition implements the parent adaptive average with deterministic CUDA gradients.
        h = h.reshape(h.size(0),h.size(1),7,4,6,4).mean(dim=(3,5))
        for i in (3,4,5,6):
            h = self.path[i](h)
        actions = self.path[7](h).split((9,2,2,2),dim=1)
        loc = F.conv2d(spatial,self.player.weight,self.player.bias).flatten(1)
        return actions, self.value(h), loc

    def forward(self, images):
        shot = torch.zeros_like(images[:,0,0,0], dtype=torch.long)
        heads, value, _ = self.outputs(images,shot)
        return heads[0], heads[1], heads[2], heads[3], value

    def locate(self, images):
        shot = torch.zeros_like(images[:,0,0,0], dtype=torch.long)
        return self.outputs(images,shot)[2]


def cell_id(xy):
    cells = (xy/8).floor().long()
    return cells[:,1].clamp(0,55)*48 + cells[:,0].clamp(0,47)


def make_schedule(seed, train_ids, source_names, arms, steps):
    rng = np.random.Generator(np.random.PCG64(seed))
    out = {arm: [] for arm in arms}
    for step in range(steps):
        rows = {arm: [] for arm in arms}
        for stage in (1,2):
            k = (stage,0)
            base = rng.choice(train_ids[k],16,replace=False)
            rows[arms[0]].extend((source_names[k],int(i)) for i in base)
            rows[arms[1]].extend((source_names[k],int(i)) for i in base[:8])
            for shot,count in ((1,2),(2,2),(4,4)):
                k = (stage,shot)
                sample = rng.choice(train_ids[k],count,replace=False)
                rows[arms[1]].extend((source_names[k],int(i)) for i in sample)
        # Same shuffling within each paired training minibatch.
        shuffle = rng.permutation(32)
        for arm in arms:
            out[arm].append([rows[arm][i] for i in shuffle])
    return out


def score(model,data,ids,device,shot=0):
    x,y,xy,changed,t=data
    preds=[];losses=[];errors=[];bomb_prob=[]
    with torch.no_grad():
        for rows in ids.split(64):
            sample=x[rows].to(device).float()/255
            if isinstance(model,ShotPolicy):
                codes = torch.full((len(rows),),shot,device=device,dtype=torch.long)
                actions,_,loc=model.outputs(sample,codes)
            else:
                assert shot==0
                actions=model(sample)[:4]
                loc=model.locate(sample)
            heads=[v.cpu() for v in actions]
            preds.append(torch.stack([v.argmax(1)for v in heads],1))
            losses.append(torch.stack([F.cross_entropy(v,y[rows,k],reduction='none')for k,v in enumerate(heads)],1))
            bomb_prob.append(heads[3].softmax(1)[:,1])
            cells=loc.argmax(1).cpu()
            errors.append((8*torch.stack((cells%48,cells//48),1)+4-xy[rows]).norm(dim=1))
    pred=torch.cat(preds);loss=torch.cat(losses);error=torch.cat(errors)
    hit=pred==y[ids];sw=changed[ids]
    return dict(n=len(ids),switches=int(sw.sum()),accuracy=hit.float().mean(0).tolist(),nll=loss.mean(0).tolist(),
                joint_nll=loss.sum(1).mean().item(),switch_move_acc=hit[sw,0].float().mean().item(),
                stable_move_acc=hit[~sw,0].float().mean().item(),switch_move_nll=loss[sw,0].mean().item(),
                stable_move_nll=loss[~sw,0].mean().item(),bomb_positive=int(y[ids,3].sum()),
                predicted_bomb_positive=int(pred[:,3].sum()),bomb_true_positive=int((pred[:,3].bool()&y[ids,3].bool()).sum()),
                mean_bomb_probability=torch.cat(bomb_prob).mean().item(),focus_positive=int(y[ids,1].sum()),
                predicted_focus_positive=int(pred[:,1].sum()),locator_median_native_px=float(np.median(error.numpy())),
                locator_p90_native_px=float(np.quantile(error.numpy(),.9))), error.numpy()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path)
    ap.add_argument('folder',type=Path)
    args=ap.parse_args()
    study=args.folder.resolve()
    protocol=study/'protocol.json'
    p=json.loads(protocol.read_bytes())
    parent=args.root/'policy/policy.ts'
    assert sha(parent)==p['parent_sha256']
    fit=study/'fit'
    fit.mkdir(exist_ok=False)
    device=torch.device('cuda')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    data={};train_ids={};names={};sanity={};splits={};offsets={};off=0
    for name,meta in sorted(p['sources'].items()):
        if meta['role']!='train':continue
        item=load_source(args.root,meta)
        data[name]=item
        ticks=item[4].numpy()
        boundary=len(ticks)*4//5
        used=np.flatnonzero(ticks[:boundary]+4 < ticks[boundary]-4)
        key=(meta['stage'],meta['shot'])
        assert key not in names
        train_ids[key]=used;names[key]=name
        sanity[name]=torch.arange(boundary,len(ticks))
        splits[name]=dict(train=len(used),sanity=len(ticks)-boundary,purged=boundary-len(used),
                          train_last_tick=int(ticks[used[-1]]),sanity_first_tick=int(ticks[boundary]))
        offsets[name]=off;off+=len(ticks)
    tensors=[torch.cat([item[k]for item in data.values()])for k in (0,1,2)]
    codes=torch.cat([torch.full((len(item[0]),),p['sources'][name]['shot'],dtype=torch.long)for name,item in data.items()])
    metadata=dict(protocol_sha256=sha(protocol),trainer_sha256=sha(Path(__file__)),parent_sha256=sha(parent),
                  gpu=torch.cuda.get_device_name(),torch=torch.__version__,splits=splits,source_offsets=offsets,pid=os.getpid())
    write(fit/'bindings.json',metadata)
    print('START',json.dumps(metadata),flush=True)
    for seed in p['seeds']:
        both=make_schedule(seed,train_ids,names,p['arms'],p['steps'])
        matrices={arm:np.array([[offsets[name]+i for name,i in batch]for batch in batches],dtype='<i8')for arm,batches in both.items()}
        for a,b in zip(*matrices.values()):
            assert len(set(a).intersection(b))==16
        np.savez_compressed(fit/f'{seed}-batches.npz',**matrices)
        for arm,matrix in matrices.items():
            folder=fit/f'{seed}-{arm}';folder.mkdir()
            torch.manual_seed(seed)
            original=torch.jit.load(str(parent),map_location=device).eval()
            model=ShotPolicy(original).to(device).eval()
            ids=torch.as_tensor(matrix[0][:8])
            test=tensors[0][ids].to(device).float()/255
            with torch.no_grad():
                diff=max(float((x-y).abs().max())for x,y in zip(model(test),original(test)))
            assert diff<3e-5,diff
            optimizer=torch.optim.Adam(model.parameters(),lr=p['lr'])
            progress=[];vals=[];start=time.monotonic()
            for step,row in enumerate(torch.from_numpy(matrix),1):
                x=tensors[0][row].to(device).float()/255
                y=tensors[1][row].to(device);pos=tensors[2][row].to(device)
                actions,_,loc=model.outputs(x,codes[row].to(device))
                action_loss=sum(F.cross_entropy(head,y[:,k])for k,head in enumerate(actions))
                loc_loss=F.cross_entropy(loc,cell_id(pos))
                optimizer.zero_grad(set_to_none=True)
                (action_loss+p['xy_weight']*loc_loss).backward()
                optimizer.step()
                vals.append([float(action_loss.detach()),float(loc_loss.detach())])
                if step%67==0:
                    block=np.mean(vals,axis=0).tolist();vals=[]
                    progress.append(dict(step=step,action_ce=block[0],xy_ce=block[1],seconds=time.monotonic()-start))
                    print(seed,arm,json.dumps(progress[-1]),flush=True)
            torch.save(model.state_dict(),folder/'conditioned.pt')
            model.cpu()
            fixture=tensors[0][ids[:3]].float()/255
            with torch.no_grad():
                runtime=torch.jit.trace_module(model,{'forward':(fixture,), 'locate':(fixture,)},check_trace=False)
                runtime.save(str(folder/'policy.ts'))
                verify=torch.jit.load(str(folder/'policy.ts')).eval()
                test_error=0.
                for n in (1,3,7):
                    hold=tensors[0][ids[:n]].float()/255
                    for l,r in zip((*model(hold),model.locate(hold)),(*verify(hold),verify.locate(hold))):
                        test_error=max(test_error,float((l-r).abs().max()))
                assert test_error<1e-5,test_error
            counts=np.bincount(codes[torch.from_numpy(matrix.flatten())].numpy(),minlength=6)
            stats=dict(seed=seed,arm=arm,steps=p['steps'],examples=int(matrix.size),shot_counts=counts.tolist(),
                       sampling_sha256=hashlib.sha256(matrix.tobytes()).hexdigest(),
                       initial_parent_max_abs=diff,export_max_abs=test_error,policy_sha256=sha(folder/'policy.ts'),
                       train_state_sha256=sha(folder/'conditioned.pt'),progress=progress)
            write(folder/'fit.json',stats)
            del original,model,optimizer,runtime,verify
    # No held source is loaded until every final actor and its training state is signed.
    held={name:load_source(args.root,m)for name,m in sorted(p['sources'].items())if m['role']=='held'}
    results={};pooled=[]
    for seed in p['seeds']:
        results[seed]={}
        for arm in p['arms']:
            folder=fit/f'{seed}-{arm}'
            stats=json.loads((folder/'fit.json').read_bytes())
            assert sha(folder/'policy.ts')==stats['policy_sha256']
            runtime=torch.jit.load(str(folder/'policy.ts'),map_location=device).eval()
            held_scores={}
            for name,item in held.items():
                entry,err=score(runtime,item,torch.arange(len(item[0])),device)
                held_scores[name]=entry
                if arm=='mixed' and name.endswith('s1'):pooled.append(err)
            trained=ShotPolicy(torch.jit.load(str(parent),map_location=device)).to(device).eval()
            trained.load_state_dict(torch.load(folder/'conditioned.pt',map_location=device,weights_only=True))
            diagnostics={name:score(trained,item,sanity[name],device,p['sources'][name]['shot'])[0]for name,item in data.items()}
            results[seed][arm]=dict(policy_sha256=stats['policy_sha256'],held=held_scores,sanity=diagnostics)
    summaries={}
    for stage in (1,2):
        pairs=[];key=f'zylo-reimua-s{stage}'
        for seed in p['seeds']:
            a,b=(results[seed][arm]['held'][key]for arm in p['arms'])
            pairs.append(dict(seed=seed,move=b['accuracy'][0]-a['accuracy'][0],switch=b['switch_move_acc']-a['switch_move_acc'],
                              stable=b['stable_move_acc']-a['stable_move_acc'],nll=b['nll'][0]-a['nll'][0],locator=b['locator_median_native_px']))
        mean={k:float(np.mean([v[k]for v in pairs]))for k in ('move','switch','stable','nll','locator')}
        summaries[key]=dict(paired=pairs,mean=mean,positive_seeds=sum(v['move']>0 for v in pairs))
    summary=summaries['zylo-reimua-s1'];avg=summary['mean'];g=p['gate']
    passed=avg['move']>=g['mean_delta_move_accuracy'] and summary['positive_seeds']>=g['positive_move_seed_pairs_min'] and avg['switch']>=g['mean_delta_changed_expert_direction_accuracy_min'] and avg['locator']<=g['locator_max_native_px']
    write(fit/'result.json',dict(**metadata,results=results,stages=summaries,passed=passed,pooled_primary_locator_median=float(np.median(np.concatenate(pooled))),gpu_peak_allocated_mb=torch.cuda.max_memory_allocated()/1e6))
    print('SEALED',summaries,passed,flush=True)


if __name__=='__main__':
    main()
