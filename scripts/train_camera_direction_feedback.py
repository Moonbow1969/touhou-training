"""Test whether the previous executed direction helps a shared human camera policy."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.checkpoint_storage import digest
from scripts.train_causal_replay_history import paths
from scripts.train_five_original_games import freeze
from scripts.train_human_camera_policy import load_replay
from scripts.train_replay_camera_encoder import CameraPolicy


class DirectionResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(137, 128)
        self.output = nn.Linear(128, 9)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features, logits, previous):
        encoded = F.one_hot(previous, 9).to(features.dtype)
        return logits + self.output(F.relu(self.hidden(torch.cat((features, encoded), dim=1))))


def score(probabilities, direction, previous):
    result = {}
    for key, mask in [('all', np.ones(len(direction), dtype=bool)), ('change', previous != direction),
                      ('start', (previous == 4) & (direction != 4)), ('turn', (previous != 4) & (direction != 4) & (previous != direction))]:
        ids = np.flatnonzero(mask)
        result[key] = dict(rows=len(ids), correct=int((probabilities[ids].argmax(1) == direction[ids]).sum()),
                           nll_sum=float(-np.log(np.maximum(probabilities[ids, direction[ids]], 1e-30)).sum()))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    d = p.parse_args().data.resolve()
    root = d/'four-title-direction-feedback'
    root.mkdir(exist_ok=True)
    parent = d/'specialist-vs-shared/merged-equal'
    split = json.loads((d/'specialist-vs-shared/protocol.json').read_bytes())
    plan = dict(train=split['train'], held=split['held'], parent=digest(parent/'policy.ts'), seed=261204,
                updates=4000, rows_per_game_rank=16, learning_rate=.0003,
                comparison='Freeze human visual features and all original heads. Fit identical residual direction heads with either previous executed direction or a constant neutral input. Inputs and RNG batches match across arms. No action resampling heuristic. Equal weight for eight game and rank cells. Deployable conditioning uses the action the agent itself issued.',
                evaluation='Repeated development players only. Report teacher-forced conditional likelihood and exact model-generated direction marginals along observed human images. Reset generated direction to neutral at each segment boundary. Other action heads remain frozen. Fixed-camera propagation does not simulate what the game would show after a different action.',
                original_test_gate='Compared to the image residual, the feedback residual must improve macro teacher-forced all-direction NLL and autonomous fixed-camera change NLL in at least three of four games, and macro autonomous change NLL must beat the frozen parent.')
    freeze(root/'protocol.json', plan)
    torch.set_num_threads(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    camera = CameraPolicy()
    state = torch.load(parent/'camera-policy.pt', map_location='cpu', weights_only=True)
    if any('policy.spatial_context.' in k for k in state):
        camera.add_spatial_context()
    camera.load_state_dict(state)
    camera.to(device).eval()
    sources = paths(d)
    stages, strata, tensors = {}, {}, []
    groups = {'train': set(), 'held': set()}
    offset = 0
    with torch.no_grad():
        for role in groups:
            for title, files in plan[role].items():
                for key in files:
                    row = load_replay(sources[key])
                    if row['sha256'] != key or row['title'] != title:
                        raise ValueError('Changed replay input')
                    groups[role].add(row['group'])
                    frame = row['native_frame']
                    direction = row['action'][:, 0]
                    previous = row['prior'][:, 0]
                    # Align both labels to the actual commands on successive captured decisions.
                    keep = np.flatnonzero((np.diff(frame) == 4) & (previous[1:] == direction[:-1])) + 1
                    if len(keep) == 0:
                        raise ValueError('Missing consecutive human decisions')
                    features = []
                    for ids in np.array_split(keep, (len(keep) + 63)//64):
                        x = torch.from_numpy(row['raw'][ids]).to(device).float()/255
                        features.append(camera.policy[:-1](camera.spatial(x)))
                    z = torch.cat(features)
                    logits = camera.policy[-1](z)[:, :9]
                    tensors.append((z, logits, torch.as_tensor(direction[keep],device=device),
                                    torch.as_tensor(previous[keep],device=device)))
                    indices = np.arange(offset, offset + len(keep))
                    stages[key] = dict(title=title, rank=row['difficulty'], role=role, player=row['group'],
                                       indices=indices, frame=frame[keep], total=len(frame))
                    if role == 'train':
                        strata.setdefault((title,row['difficulty']), []).append(indices)
                    offset += len(keep)
    if groups['train'] & groups['held'] or len(strata) != 8:
        raise ValueError('Unexpected development split')
    z, base, truth, previous = [torch.cat(parts) for parts in zip(*tensors)]
    del tensors
    banks = [torch.as_tensor(np.concatenate(strata[k]),device=device) for k in sorted(strata)]
    results = {}
    for arm in ('frozen', 'image', 'feedback'):
        torch.manual_seed(plan['seed'])
        model = DirectionResidual().to(device)
        history = []
        if arm != 'frozen':
            optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'])
            rng = torch.Generator(device=device).manual_seed(plan['seed'])
            for step in range(1, plan['updates']+1):
                ids = torch.cat([b[torch.randint(len(b),(plan['rows_per_game_rank'],),device=device,generator=rng)] for b in banks])
                source = previous[ids] if arm == 'feedback' else torch.full_like(previous[ids],4)
                loss = F.cross_entropy(model(z[ids],base[ids],source),truth[ids])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                if step==1 or step%500==0:
                    history.append(dict(step=step,nll=float(loss.detach())))
            scripted = torch.jit.script(model.eval())
            torch.testing.assert_close(scripted(z[:3],base[:3],previous[:3]),model(z[:3],base[:3],previous[:3]))
            scripted.save(str(root/(arm+'.ts')))
        results[arm] = dict(history=history, stages={})
        with torch.no_grad():
            for key, stage in stages.items():
                if stage['role'] != 'held':
                    continue
                chunks = []
                for ids in torch.as_tensor(stage['indices'],dtype=torch.long,device=device).split(96):
                    if arm == 'feedback':
                        transition = torch.stack([model(z[ids],base[ids],torch.full_like(ids,k)).softmax(1) for k in range(9)],1)
                    else:
                        logits = base[ids] if arm=='frozen' else model(z[ids],base[ids],torch.full_like(ids,4))
                        transition = logits.softmax(1)[:,None,:].expand(-1,9,-1)
                    chunks.append(transition.cpu().numpy())
                matrix = np.concatenate(chunks).astype(np.float64)
                matrix /= matrix.sum(axis=2,keepdims=True)
                ids = torch.as_tensor(stage['indices'],device=device)
                human = truth[ids].cpu().numpy()
                prior = previous[ids].cpu().numpy()
                marginal = np.empty((len(human),9))
                for i,frame in enumerate(stage['frame']):
                    if i==0 or frame-stage['frame'][i-1]!=4:
                        marginal[i] = matrix[i,4]
                    else:
                        marginal[i] = marginal[i-1] @ matrix[i]
                np.testing.assert_allclose(marginal.sum(axis=1),1.,atol=1e-10,rtol=0)
                results[arm]['stages'][key] = dict(conditional=score(matrix[np.arange(len(human)),prior],human,prior),
                                                   autonomous=score(marginal,human,prior))
        print('Scored',arm,flush=True)
    by_title = {}
    for title, files in plan['held'].items():
        by_title[title] = {}
        for arm, result in results.items():
            metrics = {}
            for mode in ('conditional','autonomous'):
                metrics[mode] = {}
                for subset in ('all','change','start','turn'):
                    rows = [result['stages'][k][mode][subset] for k in files]
                    n = sum(r['rows'] for r in rows)
                    metrics[mode][subset] = dict(n=n,nll=sum(r['nll_sum'] for r in rows)/n,accuracy=sum(r['correct'] for r in rows)/n)
            by_title[title][arm] = metrics
    def difference(title,control,mode,subset):
        return title['feedback'][mode][subset]['nll'] - title[control][mode][subset]['nll']
    changed = sum(difference(t,'image','autonomous','change')<0 for t in by_title.values())
    conditional = np.mean([difference(t,'image','conditional','all') for t in by_title.values()])
    transfer = np.mean([difference(t,'frozen','autonomous','change') for t in by_title.values()])
    report = dict(registration=digest(root/'protocol.json'), by_title=by_title, outcomes=results,
                  split={k:dict(title=r['title'],rank=r['rank'],role=r['role'],group=r['player'],usable=len(r['indices']),available=r['total']) for k,r in stages.items()},
                  model_sha256={arm:digest(root/(arm+'.ts')) for arm in ('image','feedback')},
                  autonomous_change_improved_titles=changed, macro_conditional_delta=float(conditional),
                  macro_autonomous_change_delta_vs_parent=float(transfer), original_test_gate=bool(changed>=3 and conditional<0 and transfer<0))
    (root/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print('Completed direction-feedback comparison',changed,conditional,transfer,report['original_test_gate'],flush=True)


if __name__ == '__main__':
    main()
