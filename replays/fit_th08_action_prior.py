"""Continue the four-title human action prior on the frozen TH08 cohort."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from replays.action_prior import ActionPrior, VOCAB_FIELDS, batches, tile
from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze


def pack(rows, directory):
    spans = []
    parts = []
    offset = 0
    stage_numbers = {'1': 1, '2': 2, '3': 3, '4A': 4, '4B': 4, '5': 5, '6B': 6}
    for row in rows:
        path = directory / (Path(row['replay']).stem + '-actions.npz')
        with np.load(path, allow_pickle=False) as archive:
            for stage, expected in row['stages'].items():
                actions = archive[stage]
                if (actions.shape != (expected['decisions'], 4)
                        or hashlib.sha256(actions.tobytes()).hexdigest() != expected['action_sha256']):
                    raise ValueError('Decoded original inputs have changed')
                tokens = (actions @ np.array([1, 9, 18, 36])).astype(np.uint8)
                parts.append(tokens)
                spans.append(dict(author=row['author'], name=row['replay'], title=8, difficulty=row['rank'],
                                  shot=row['team'], stage=stage_numbers[stage], start=offset,
                                  length=len(tokens), split=row['split']))
                offset += len(tokens)
    return np.concatenate(parts), spans


@torch.inference_mode()
def score(model, stream, spans, vocab, device):
    model.eval()
    result = {}
    for author in sorted({s['author'] for s in spans if s['split'] == 'held'}):
        subset = [s for s in spans if s['author'] == author]
        group = tile(subset, 'held', 64, vocab)
        counts = np.zeros(6, dtype=np.int64)
        nll = 0.
        for keys, targets, labels, positions in batches(stream, group, np.arange(len(group[0])), device, 64, 128):
            logits = model(keys, labels, positions)
            guess = logits.reshape(*targets.shape, 8, 9).logsumexp(-2).argmax(-1)
            real = targets % 9
            changed = keys % 9 != real
            correct = guess == real
            counts += [targets.numel(), int(correct.sum()), int(changed.sum()), int((changed & correct).sum()),
                       int((logits.argmax(-1) == targets).sum()), int((targets == keys).sum())]
            nll += float(F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction='sum'))
        n, match, changes, turns, exact, same = (int(x) for x in counts)
        result[author] = dict(rank=subset[0]['difficulty'], team=subset[0]['shot'], n=n, changes=changes,
                              next_direction=match/n, repeat_direction=(n-changes)/n,
                              correct_changes=turns/changes, joint_action_accuracy=exact/n,
                              repeated_joint_action=same/n, joint_nll=nll/n)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'cache', 'parent'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    manifest = args.source / 'source-manifest.json'
    source = json.loads(manifest.read_bytes())
    out = args.source / 'action-prior'
    out.mkdir(exist_ok=True)
    weights = args.parent / 'prior.pt'
    if digest(weights) != '22981e8b48e829b57730e1723ad0d93d4051e9154f7a846f6e67f8446348dfee':
        raise ValueError('Expected the independently archived four-title H100 model')
    plan = dict(source_manifest_sha256=digest(manifest), pretrained_weights_sha256=digest(weights),
                pretrained_vocabulary_sha256=digest(args.parent / 'vocab.json'), seed=261128, epochs=4,
                batch_sequences=128, sequence_decisions=64, optimizer='AdamW lr=0.0002',
                training='All complete nonoverlapping 64-decision tiles, restarts at stage boundaries with a preceding '
                         'human command. Single shuffled pass per epoch. Original four-title vocabulary uses its '
                         'reserved unknown ID for TH08 and each of its four new teams. No validation-based selection.',
                scoring='Score frozen, one-pass and four-pass models once on the same separately assigned TH08 '
                        'players. Report per-person direction, actual changes, joint four-head token accuracy and '
                        'joint NLL. Primary comparison is four passes minus frozen, macro-averaged across authors. '
                        'Only earlier genuine human keys and metadata are available. Original images are absent.')
    freeze(out / 'protocol.json', plan)
    rows = source['qualified']
    if len(rows) != 64 or source['excluded'] or len({r['author'] for r in rows}) != 64:
        raise ValueError('The registered complete TH08 population is required for this fit')
    stream, spans = pack(rows, args.cache)
    vocab = json.loads((args.parent / 'vocab.json').read_bytes())
    vocab['stage'] = {int(k): v for k, v in vocab['stage'].items()}
    vocab['title'] = {int(k): v for k, v in vocab['title'].items()}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(plan['seed'])
    torch.set_num_threads(2)
    model = ActionPrior([len(vocab[k])+1 for k in VOCAB_FIELDS]).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    train = tile(spans, 'train', 64, vocab)
    rng = np.random.default_rng(plan['seed'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scores = {'frozen': score(model, stream, spans, vocab, device)}
    training = []
    for epoch in range(1, 5):
        model.train()
        total_loss = 0.
        total = 0
        for keys, targets, meta, pos in batches(stream, train, rng.permutation(len(train[0])), device, 64, 128):
            optimizer.zero_grad(set_to_none=True)
            logits = model(keys, meta, pos)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * targets.numel()
            total += targets.numel()
        training.append(dict(epoch=epoch, presentations=total, mean_train_joint_nll=total_loss/total))
        (out / 'training.json').write_text(json.dumps(training, indent=2) + '\n')
        if epoch in (1, 4):
            torch.save(model.state_dict(), out / f'epoch-{epoch}.pt')
            scores[f'epoch-{epoch}'] = score(model, stream, spans, vocab, device)
        print(training[-1], flush=True)
    metrics = ('next_direction', 'correct_changes', 'joint_action_accuracy', 'joint_nll')
    summary = {}
    for rank in ('Hard', 'Lunatic', 'All'):
        authors = [k for k, s in scores['frozen'].items() if rank == 'All' or s['rank'] == rank]
        by_rank = dict(authors=len(authors), decisions=sum(scores['frozen'][a]['n'] for a in authors))
        for field in metrics:
            measures = {arm: np.array([values[a][field] for a in authors]) for arm, values in scores.items()}
            deltas = measures['epoch-4'] - measures['frozen']
            draws = np.random.default_rng(plan['seed']).choice(deltas, (10000, len(authors)), replace=True)
            by_rank[field] = dict(means={a: float(x.mean()) for a, x in measures.items()},
                                  paired_gain=float(deltas.mean()),
                                  author_bootstrap95=np.quantile(draws.mean(1), [.025,.975]).tolist())
        summary[rank] = by_rank
    freeze(out / 'report.json', dict(protocol_sha256=digest(out / 'protocol.json'),
        device=torch.cuda.get_device_name() if device == 'cuda' else 'CPU', unique_training_tiles=len(train[0]),
        unique_training_decisions=64*len(train[0]), training=training, per_author=scores, macro=summary,
        models={str(e): digest(out / f'epoch-{e}.pt') for e in (1,4)}))


if __name__ == '__main__':
    main()
