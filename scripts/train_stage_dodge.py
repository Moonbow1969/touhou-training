import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from scripts.evaluate_stage_policies import read_policy
from scripts.pretrain_policy import append_motion_history, batch_block, condition_context, index_split, predict_heads, read_shard
from scripts.stage_risk import action_clearances, suggest_dodge


def candidate_features(action, clearances):
    move, focus = map(int, action[:2])
    margin = clearances.ravel()
    selected = 9 * focus + move
    ids = np.arange(18)
    current = margin[selected]
    full = np.full_like(margin, np.clip(current, -16, 32) / 16)
    return np.stack((np.clip(margin, -16, 32) / 16, full,
                     np.clip(margin - current, -32, 32) / 16,
                     ids == selected, ids // 9 == focus), axis=-1).astype(np.float32)


class DodgeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(5, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, features):
        return self.score(features).squeeze(-1)


def learned_action(model, action, clearances):
    with torch.inference_mode():
        scores = model(torch.from_numpy(candidate_features(action, clearances)))
    focus, move = divmod(int(scores.argmax()), 9)
    proposal = action.copy()
    proposal[:2] = (move, focus)
    return proposal


def sample_states(directory, split, maximum, rng, policy, device):
    shards = [read_shard((path, meta), True) for path, meta in index_split(directory, split) if meta['title'] == 14]
    if not shards:
        raise ValueError(f'No TH14 replay data in {split}.')
    combined = tuple(np.concatenate(fields) for fields in zip(*shards))
    selection = rng.choice(len(combined[2]), min(maximum, len(combined[2])), replace=False)
    for bullets, contexts, _ in batch_block(combined, 256, device, selection):
        with torch.inference_mode():
            logits = predict_heads(policy, bullets, contexts)
            actions = torch.stack([head.argmax(dim=1) for head in logits], dim=1).cpu().numpy()
        yield from zip(bullets.cpu().numpy(), contexts.cpu().numpy(), actions)


def examples(states, rng):
    result = []
    for bullets, context, action in states:
        clearance = action_clearances({'bullets': bullets, 'context': context}, 1.0)
        randomized = action.copy()
        randomized[:2] = (rng.integers(9), rng.integers(2))
        for prior in (action, randomized):
            target, _, _ = suggest_dodge(prior, clearance)
            result.append((candidate_features(prior, clearance), 9 * target[1] + target[0],
                           bool((target[:2] != prior[:2]).any())))
    return result


def demonstrated_corrections(path):
    result = []
    with np.load(path) as data:
        for bullets, context, prior, target in zip(data['bullets'], data['context'],
                                                  data['original_action'], data['action']):
            clearance = action_clearances({'bullets': bullets, 'context': context}, 1.0)
            result.append((candidate_features(prior, clearance), 9 * target[1] + target[0], True))
    return result


def native_examples(paths, policy):
    result = []
    for path in paths:
        with np.load(path) as data:
            context = append_motion_history(condition_context(data['context'], 1, 'ReimuA'), data['stage_frame'])
            proposals = policy.predict({'bullets': data['bullets'], 'context': context}, deterministic=True)[0]
            clearances = data['action_clearances'].min(axis=1)
        for proposal, clearance in zip(proposals, clearances):
            target, _, _ = suggest_dodge(proposal, clearance)
            result.append((candidate_features(proposal, clearance), 9 * target[1] + target[0],
                           bool((target[:2] != proposal[:2]).any())))
    return result


def dataset(rows):
    return TensorDataset(*(torch.from_numpy(np.stack(field)) for field in zip(*rows)))


def clearance_targets(features, labels, mixture):
    hard = F.one_hot(labels, num_classes=features.shape[1]).float()
    margins = features[..., 0] * 16
    selected = margins.gather(1, labels[:, None])
    eligible = ((margins >= 0) & (margins >= selected - 3)) | hard.bool()
    alternative = torch.softmax((margins / 4).masked_fill(~eligible, -torch.inf), dim=1)
    return (1 - mixture) * hard + mixture * alternative


@torch.inference_mode()
def metrics(model, examples, mixture):
    inputs, labels, changed = examples
    predictions = model(inputs)
    correct = predictions.argmax(1) == labels
    targets = clearance_targets(inputs, labels, mixture) if mixture else labels
    distribution = torch.distributions.Categorical(logits=predictions)
    return {'decisions': len(labels), 'corrections': int(changed.sum()),
            'validation_loss': float(F.cross_entropy(predictions, targets)),
            'all_accuracy': float(correct.float().mean()),
            'mean_entropy': float(distribution.entropy().mean()),
            'negative_clearance_probability': float((distribution.probs * (inputs[..., 0] < 0)).sum(1).mean()),
            'correction_accuracy': float(correct[changed].float().mean()) if changed.any() else None}


def main():
    parser = argparse.ArgumentParser(description='Train a neural action selector from replay and closed-loop dodge corrections.')
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--corrections', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--clearance-mixture', type=float, default=0.,
                        help='Preserve probability on nearly as safe candidate actions.')
    parser.add_argument('--native-train', type=Path, nargs='+', default=[])
    parser.add_argument('--native-test', type=Path, nargs='+', default=[])
    args = parser.parse_args()
    if args.epochs < 1 or not 0 <= args.clearance_mixture <= 1:
        parser.error('Epochs must be positive and the clearance mixture must be between zero and one.')
    if {path.resolve() for path in args.native_train} & {path.resolve() for path in args.native_test}:
        parser.error('Use separate native captures for training and testing.')
    torch.manual_seed(203)
    torch.set_num_threads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.default_rng(203)
    base = read_policy(args.base, device, motion_history=True)
    if (args.native_train or args.native_test) and (base.stage_clock or base.motion_spans):
        parser.error('Native capture training currently expects the immediate-history TH14 Reimu A base.')
    replay_train = examples(sample_states(args.data, 'train', 4500, rng, base, device), rng)
    corrections = demonstrated_corrections(args.corrections)
    capture_train = native_examples(args.native_train, base)
    replay_validation = examples(sample_states(args.data, 'validation', 700, rng, base, device), rng)
    train = dataset(replay_train + corrections * 12 + capture_train)
    validation = dataset(replay_validation)
    validation_data = tuple(t.to(device) for t in validation.tensors)
    model = DodgeHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    args.out.mkdir(parents=True, exist_ok=False)
    best = float('inf')
    epochs = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y, changed_action in DataLoader(train, batch_size=192, shuffle=True):
            x, y = x.to(device), y.to(device)
            weights = 1 + changed_action.to(device).float()
            labels_for_loss = clearance_targets(x, y, args.clearance_mixture) if args.clearance_mixture else y
            loss = (F.cross_entropy(model(x), labels_for_loss, reduction='none') * weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        row = {'epoch': epoch, **metrics(model, validation_data, args.clearance_mixture)}
        epochs.append(row)
        if row['validation_loss'] < best:
            best = row['validation_loss']
            torch.save(model.state_dict(), args.out / 'head.pt')
    report = {'replay_training_decisions': len(replay_train), 'on_policy_corrections': len(corrections),
              'clearance_mixture': args.clearance_mixture,
              'native_training_decisions': len(capture_train),
              'validation_decisions': len(validation), 'validation_corrections': int(validation_data[2].sum()),
              'best': min(epochs, key=lambda row: row['validation_loss']), 'epochs': epochs}
    if args.native_test:
        holdout = dataset(native_examples(args.native_test, base))
        model.load_state_dict(torch.load(args.out / 'head.pt', map_location=device, weights_only=True))
        report['native_test'] = metrics(model, tuple(t.to(device) for t in holdout.tensors), args.clearance_mixture)
    (args.out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print({key: value for key, value in report.items() if key != 'epochs'}, flush=True)


if __name__ == '__main__':
    main()
