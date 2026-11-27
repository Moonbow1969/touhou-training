"""Fit matched human direction heads using an expert prior or a propagated predicted prior."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.checkpoint_storage import digest
from scripts.export_camera_direction_feedback import CameraWithDirection
from scripts.train_camera_direction_feedback import DirectionResidual, score
from scripts.train_five_original_games import freeze
from scripts.train_human_camera_policy import load_replay
from scripts.train_replay_camera_encoder import CameraPolicy


def read(path):
    return json.loads(path.read_bytes())


def transitions(model, features, base):
    states = torch.arange(9, device=features.device).expand(len(features), 9)
    def repeat(tensor):
        return tensor[:, None].expand(-1, 9, -1).reshape(len(features) * 9, -1)
    return model(repeat(features), repeat(base), states.reshape(-1)).reshape(-1, 9, 9)


def encode(camera, file, device):
    row = load_replay(file)
    with torch.no_grad():
        latent = torch.cat([camera.policy[:-1](camera.spatial(px.to(device).float() / 255))
                            for px in torch.from_numpy(row['raw']).split(64)])
        base = camera.policy[-1](latent)[:, :9].contiguous()
    return row, latent, base


def assess(model, camera, roster, d, device):
    output = []
    with torch.no_grad():
        for source in roster:
            path = d / source['path']
            if digest(path) != source['archive_sha256'] or source['split'] != 'held':
                raise ValueError('Unverified held source')
            row, z, base = encode(camera, path, device)
            matrices = torch.cat([transitions(model, z[i:i+96], base[i:i+96]).softmax(2).cpu()
                                  for i in range(0, len(z), 96)]).double().numpy()
            matrices /= matrices.sum(2, keepdims=True)
            direction = row['action'][:, 0]
            marginal = np.empty((len(z), 9))
            changes = reversals = 0.
            pairs = 0
            for i, matrix in enumerate(matrices):
                if i == 0 or row['native_frame'][i] - row['native_frame'][i - 1] != 4:
                    marginal[i] = matrix[4]
                    continue
                joint = marginal[i - 1][:, None] * matrix
                marginal[i] = joint.sum(0)
                changes += 1 - np.trace(joint)
                reversals += np.trace(joint[:, ::-1]) - joint[4, 4]
                pairs += 1
            np.testing.assert_allclose(marginal.sum(1), 1., atol=1e-9, rtol=0.)
            output.append(dict(player=row['group'], title=row['title'], rank=row['difficulty'], sha256=row['sha256'],
                               sampled_changes=changes, sampled_reversals=reversals, adjacent_pairs=pairs,
                               predicted_on_own_previous_action=score(marginal, direction, row['prior'][:, 0])))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--prospective', action='store_true')
    args = parser.parse_args()
    d = args.data.resolve()
    out = d / 'four-title-four-step-feedback'
    out.mkdir(exist_ok=True)
    parent = d / 'four-title-later-stage-sft/policy'
    manifest = d / 'human-train-six-complete-credits/later-train-first-21-frozen.json'
    held = d / 'human-retail17-wave2-held/held-original-synchronized-frozen.json'
    roster = d / 'unseen-later-stage-author-challenge/prospective-six-players-frozen.json'
    rows = read(manifest)
    files = [d / v['path'] for v in rows['prior47_train'] + rows['qualified']]
    parent_report = read(parent / 'report.json')
    actors = ('teacher', 'propagated')
    plan = dict(parent=digest(parent / 'policy.ts'), source_manifest=digest(manifest), repeated_held=digest(held),
                future_held=digest(roster), seed=1781101, updates=1600, starts_per_game_rank=8, horizon=4, lr=.0003,
                design='Freeze all 68-stage SFT camera parameters. Both residuals receive identical four-command subsequences, equal exposure over four titles and both ranks, optimizer, initialization, model and four CE losses. Teacher conditions every step on the actual preceding human action. Propagated starts from that same human action at step one, then uses its own exact nine-state marginal along subsequent expert screens. Both deployment models condition only on their own previous chosen direction and current image.',
                outcome='Primary prospective outcome is the mean across eligible new pilots of mean start/turn NLL using own previous predicted directions on fixed expert images. Report changes, starts, turns, sampled reversals and all-command NLL, both on repeated TH17 and the prospective six TH10/12/13 pilots. The movement stats under fixed expert screens estimate control behavior only.')
    freeze(out / 'protocol.json', plan)
    if plan['parent'] != parent_report['torchscript_sha256'] or digest(parent / 'camera-policy.pt') != parent_report['sft'] or {digest(f) for f in files} != {v['sha256'] for v in parent_report['train']}:
        raise ValueError('Changed human training inputs or encoder')
    torch.set_num_threads(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    camera = CameraPolicy()
    camera.load_state_dict(torch.load(parent / 'camera-policy.pt', map_location='cpu', weights_only=True))
    camera.to(device).eval()
    if args.prospective:
        final = read(roster.with_name('held-original-synchronized-frozen.json'))
        allocated = {(v['sha256'], v['reserved_stage']) for v in read(roster)['sources']}
        outcomes = final['qualified'] + final['excluded']
        if len(outcomes) != 6 or {(v['replay_sha256'], v['stage']) for v in outcomes} != allocated:
            raise ValueError('Missing future allocation')
        sources = final['qualified']
    else:
        sources = read(held)['qualified']
    if {v['player_group'].casefold() for v in sources} & {v['group'] for v in parent_report['train']}:
        raise ValueError('Validation player in training')
    if not args.prospective and not all((out / name / 'fit.json').is_file() for name in actors):
        pieces = []
        starts = defaultdict(list)
        offset = 0
        for path in files:
            row, latent, base = encode(camera, path, device)
            expert = row['action'][:, 0]
            prior = row['prior'][:, 0]
            continuous = (np.diff(row['native_frame']) == 4) & (prior[1:] == expert[:-1])
            valid = np.flatnonzero(np.lib.stride_tricks.sliding_window_view(continuous, 3).all(1))
            starts[row['title'], row['difficulty']].append(valid + offset)
            pieces.append((latent, base, torch.as_tensor(expert.copy(), device=device), torch.as_tensor(prior.copy(), device=device)))
            offset += len(expert)
        if len(starts) != 8 or offset != 231511:
            raise ValueError('Incomplete training bank')
        banks = [torch.as_tensor(np.concatenate(starts[key]), device=device) for key in sorted(starts)]
        features, logits, expert, prior = [torch.cat(part) for part in zip(*pieces)]
        del pieces
        for name in actors:
            dest = out / name
            dest.mkdir(exist_ok=True)
            if (dest / 'fit.json').exists():
                continue
            torch.manual_seed(plan['seed'])
            rng = torch.Generator(device=device).manual_seed(plan['seed'])
            model = DirectionResidual().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=plan['lr'])
            trace = []
            for step in range(1, plan['updates'] + 1):
                first = torch.cat([bank[torch.randint(len(bank), (8,), device=device, generator=rng)] for bank in banks])
                ids = first[:, None] + torch.arange(4, device=device)
                batch = ids.flatten()
                matrix = transitions(model, features[batch], logits[batch]).log_softmax(2).reshape(len(first), 4, 9, 9)
                ix = torch.arange(len(first), device=device)
                dist = F.one_hot(prior[first], 9).to(features.dtype)
                terms = []
                for j in range(4):
                    if name == 'teacher':
                        terms.append(-matrix[ix, j, prior[ids[:, j]], expert[ids[:, j]]].mean())
                    else:
                        dist = (dist[:, :, None] * matrix[:, j].exp()).sum(1)
                        terms.append(-dist[ix, expert[ids[:, j]]].clamp_min(1e-20).log().mean())
                loss = torch.stack(terms).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                if step == 1 or step % 200 == 0:
                    trace.append(dict(update=step, loss=float(loss.detach())))
            residual = torch.jit.script(model.eval())
            residual.save(str(dest / 'residual.ts'))
            actor = CameraWithDirection(camera, model, True).eval()
            fixture = torch.rand(9, 2, 112, 96, device=device)
            choices = torch.arange(9, device=device)
            with torch.no_grad():
                export = torch.jit.trace(actor, (fixture[:1], choices[:1]))
                for actual, compiled in zip(actor(fixture, choices), export(fixture, choices)):
                    torch.testing.assert_close(actual, compiled)
                for parent_head, fitted_head in zip(camera(fixture)[1:], export(fixture, choices)[1:]):
                    torch.testing.assert_close(parent_head, fitted_head)
            export.save(str(dest / 'policy.ts'))
            freeze(dest / 'fit.json', dict(registration=digest(out / 'protocol.json'), residual=digest(dest / 'residual.ts'),
                   actor=digest(dest / 'policy.ts'), available_four_decision_starts={str(key):sum(map(len, group)) for key, group in starts.items()}, history=trace))
            print('Fitted', name, digest(dest / 'policy.ts'), flush=True)
    for name in actors:
        dest = out / name
        record = read(dest / 'fit.json')
        if digest(dest / 'policy.ts') != record['actor'] or digest(dest / 'residual.ts') != record['residual']:
            raise ValueError('Frozen head changed')
        model = torch.jit.load(str(dest / 'residual.ts'), map_location=device).eval()
        result = assess(model, camera, sources, d, device)
        freeze(dest / ('prospective-six.json' if args.prospective else 'repeated-th17.json'), result)


if __name__ == '__main__':
    main()
