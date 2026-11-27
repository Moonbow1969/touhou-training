"""Measure replay retention when extending a four-title action prior to TH08 at fixed compute."""
import argparse

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from replays.action_prior import ActionPrior, VOCAB_FIELDS, batches, tile
from replays.audit_th08_transfer import old_credit_lookup
from replays.fit_th08_action_prior import pack, score
from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze


def roster(args):
    study = args.data / 'th08-human-actions'
    new = json.loads((study / 'source-manifest.json').read_bytes())['qualified']
    new_stream, new_spans = pack(new, args.cache)
    audit = json.loads((args.data / 'human-train-incremental/action-prior-vs-camera-credited-author-audit.json').read_bytes())
    stages_path = args.tokens / 'stages.json'
    if digest(stages_path) != audit['historical_action_only_stage_manifest_sha256']:
        raise ValueError('Historical stream index changed')
    old_stream = np.load(args.tokens / 'tokens.npy', mmap_mode='r')
    old = json.loads(stages_path.read_bytes())
    authors = old_credit_lookup(args.historical, audit)
    old_train = {authors[s['name']] for s in old if s['split'] == 'train'}
    new_train = {s['author'] for s in new if s['split'] == 'train'}
    new_held = {s['author'] for s in new if s['split'] == 'held'}
    train, held = list(s for s in new_spans if s['split'] == 'train'), list(s for s in new_spans if s['split'] == 'held')
    for s in old:
        if s['difficulty'] not in ('Hard', 'Lunatic'):
            continue
        author = authors[s['name']]
        row = dict(s, author=author, start=int(s['start']) + len(new_stream))
        if s['split'] == 'train' and author not in new_held:
            train.append(row)
        if s['split'] == 'test' and author not in old_train | new_train:
            held.append(dict(row, split='held'))
    if {s['author'] for s in train} & {s['author'] for s in held}:
        raise ValueError('Shared credited person across continuation train and held partitions')
    sources = {part: [dict(s) for s in seq] for part,seq in [('train',train),('held',held)]}
    for seq in sources.values():
        for s in seq:
            s.pop('start')
    return np.concatenate((new_stream,old_stream)), train, held, sources


def evaluate(model, stream, held, vocab, device, *, turn_detection=False):
    result = {}
    for title in (8,13,14,16,17):
        for rank in ('Hard','Lunatic'):
            rows = [s for s in held if s['title'] == title and s['difficulty'] == rank]
            if rows:
                result[f'TH{title:02}-{rank}'] = score(model,stream,rows,vocab,device,turn_detection=turn_detection)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data','tokens','cache','historical','parent'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    out = args.data / 'five-title-human-action-rehearsal'
    out.mkdir(exist_ok=True)
    stream,train,held,sources = roster(args)
    freeze(out/'sources.json',sources)
    data = args.data/'th08-human-actions/action-prior'
    single = json.loads((data/'report.json').read_bytes())
    parent = args.parent/'prior.pt'
    previous_plan = json.loads((data/'protocol.json').read_bytes())
    if digest(parent) != previous_plan['pretrained_weights_sha256'] or digest(data/'epoch-4.pt') != single['models']['4']:
        raise ValueError('Expected common initial weights and the TH08-only comparator')
    vocab = json.loads((args.parent/'vocab.json').read_bytes())
    for key in ('title','stage'):
        vocab[key] = {int(k):v for k,v in vocab[key].items()}
    seed, epochs, length = previous_plan['seed'], 4, 64
    count = single['unique_training_tiles']
    rows = tile(train,'train',length,vocab)
    cells = [(t,r) for t in (8,13,14,16,17) for r in ('Hard','Lunatic')]
    pools = [np.flatnonzero((rows[1][:,0] == vocab['title'].get(t,0)) &
                            (rows[1][:,1] == vocab['difficulty'][r])) for t,r in cells]
    if sum(map(len,pools)) != len(rows[0]):
        raise ValueError('Tile accounting failed')
    plan = dict(seed=seed, parent_sha256=digest(parent), th08_only_sha256=single['models']['4'],
                source_sha256=digest(out/'sources.json'), token_array_sha256=digest(args.tokens/'tokens.npy'),
                epochs=epochs, tiles_per_epoch=count, decisions_per_tile=length, batch_sequences=128,
                optimizer='AdamW lr=0.0002', cells=[dict(title=t,rank=r,eligible_tiles=len(q)) for (t,r),q in zip(cells,pools)],
                sampling='Half the tile budget to TH08, divided evenly between ranks. Half to the other eight cells equally. Rotate integer remainders per epoch. Draw without replacement in each cell and shuffle across cells.',
                comparisons='Fixed 4-pass TH08-only and mixed models from exact same initial weights with equal exposures. Frozen mean-metadata control uses mean trained title and shot rows for unknown metadata. Metrics by held credited author, game and difficulty. Historical test names absent from either training source. Two TH08 held names known to parent weights.',
                primary='Direction accuracy on actual turns. Report next-direction accuracy and joint NLL alongside. Historical TH16 Hard has zero independently held files.')
    freeze(out/'protocol.json',plan)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=ActionPrior([len(vocab[k])+1 for k in VOCAB_FIELDS]).to(device)
    model.load_state_dict(torch.load(parent,map_location=device,weights_only=True))
    rng=np.random.default_rng(seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4)
    logs=[]
    all_seen=set()
    for epoch in range(epochs):
        new_budget = (count + epoch%2)//2
        old_budget = count-new_budget
        budgets=[new_budget//2]*2+[old_budget//8]*8
        if new_budget%2:
            budgets[epoch%2]+=1
        for j in range(old_budget%8):
            budgets[2+(epoch+j)%8]+=1
        take=np.concatenate([rng.choice(pool,n,replace=False) for pool,n in zip(pools,budgets)])
        indices=rng.permutation(take)
        if len(set(indices.tolist())) != count:
            raise ValueError('Expected unique sample within epoch')
        all_seen.update(indices.tolist())
        total_loss=0.
        model.train()
        for keys,target,meta,position in batches(stream,rows,indices,device,length,128):
            optimizer.zero_grad(set_to_none=True)
            logits=model(keys,meta,position)
            loss=F.cross_entropy(logits.flatten(0,1),target.flatten())
            loss.backward()
            optimizer.step()
            total_loss+=float(loss.detach())*target.numel()
        logs.append(dict(epoch=epoch+1, tiles_by_cell=budgets, presentations=count*length,
                         distinct_training_decisions_to_date=len(all_seen)*length,
                         sampled_tile_index_sha256=hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),
                         mean_joint_nll=total_loss/(count*length)))
        print(logs[-1],flush=True)
        (out/'training.json').write_text(json.dumps(logs,indent=2)+'\n',encoding='utf-8')
    final=out/'epoch-4.pt'
    torch.save(model.state_dict(),final)
    scores={'mixed':evaluate(model,stream,held,vocab,device)}
    model.load_state_dict(torch.load(data/'epoch-4.pt',map_location=device,weights_only=True))
    scores['th08_only']=evaluate(model,stream,held,vocab,device)
    model.load_state_dict(torch.load(parent,map_location=device,weights_only=True))
    with torch.no_grad():
        for field in ('title','shot'):
            w=model.metadata[VOCAB_FIELDS.index(field)].weight
            w[0].copy_(w[1:].mean(0))
    scores['frozen_mean']=evaluate(model,stream,held,vocab,device)
    summary={}
    for key,people in scores['mixed'].items():
        metrics={}
        for field in ('correct_changes','next_direction','joint_nll'):
            values={arm:np.array([scores[arm][key][n][field] for n in people]) for arm in scores}
            contrasts={}
            for base in ('th08_only','frozen_mean'):
                delta=values['mixed']-values[base]
                ci=np.random.default_rng(261130).choice(delta,(10000,len(delta)),replace=True).mean(1)
                contrasts[base]=dict(mean=float(delta.mean()),author_bootstrap95=np.quantile(ci,[.025,.975]).tolist() if len(delta)>1 else None)
            metrics[field]=dict(means={arm:float(s.mean()) for arm,s in values.items()},paired_mixed_minus=contrasts)
        summary[key]=dict(authors=len(people),decisions=sum(s['n'] for s in people.values()),metrics=metrics)
    freeze(out/'report.json',dict(protocol_sha256=digest(out/'protocol.json'),final_sha256=digest(final),
           device=str(torch.cuda.get_device_name() if device=='cuda' else device),summary=summary,by_author=scores))


if __name__=='__main__':
    main()
