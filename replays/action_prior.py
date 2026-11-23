import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


TOKEN_COUNT = 72
VOCAB_FIELDS = ('title', 'difficulty', 'shot', 'stage')


def pack(corpus, out):
    from scripts.pretrain_policy import label_actions

    rows = json.loads(corpus.read_text(encoding='utf-8'))
    out.mkdir(parents=True, exist_ok=False)
    size = sum(frames // 4 for row in rows for frames in row['frames_by_stage'].values())
    stream = np.lib.format.open_memmap(out / 'tokens.npy', mode='w+', dtype=np.uint8, shape=(size,))
    spans = []
    start = 0
    for row in rows:
        with np.load(corpus.parent / f"{Path(row['name']).stem}-actions.npz") as archive:
            for stage, frames in row['frames_by_stage'].items():
                words = archive[f'stage_{stage}_input_words']
                if len(words) != frames:
                    raise ValueError(f"Action count differs from manifest for {row['name']} stage {stage}.")
                windows = len(words) // 4
                actions = label_actions(words[:windows * 4].reshape(-1, 4))
                stream[start:start + windows] = actions @ np.array([1, 9, 18, 36], dtype=np.int64)
                meta = {k: row[k] for k in ('name', 'title', 'difficulty', 'shot', 'split')}
                spans.append(dict(meta, stage=int(stage), start=start, length=windows))
                start += windows
    if start != size:
        raise RuntimeError('The archive frame inventory changed during packing.')
    stream.flush()
    (out / 'stages.json').write_text(json.dumps(spans), encoding='utf-8')
    print(dict(Counter({split: sum(s['length'] for s in spans if s['split'] == split)
                        for split in ('train', 'validation', 'test')})), flush=True)


def tile(spans, split, sequence, vocab):
    starts, conditions, stage_positions = [], [], []
    for stage in spans:
        if stage['split'] != split:
            continue
        positions = np.arange(1, stage['length'] - sequence + 1, sequence)
        if not len(positions):
            continue
        condition = [vocab[key].get(stage[key], 0) for key in VOCAB_FIELDS]
        starts.append(stage['start'] + positions)
        conditions.append(np.broadcast_to(condition, (len(positions), len(condition))))
        stage_positions.append(positions)
    return tuple(np.concatenate(values) for values in (starts, conditions, stage_positions))


def batches(stream, groups, indices, device, length, batch_size):
    starts, cond, positions = groups
    offsets = np.arange(length)
    for start in range(0, len(indices), batch_size):
        take = indices[start:start + batch_size]
        ids = starts[take, None] + offsets
        position = np.log1p((positions[take, None] + offsets) / 15).astype(np.float32) / 8
        yield tuple(torch.from_numpy(value).to(device) for value in
                    (stream[ids - 1].astype(np.int64), stream[ids].astype(np.int64), cond[take], position))


class ActionPrior(nn.Module):
    def __init__(self, vocab_sizes, width=64):
        super().__init__()
        self.keys = nn.Embedding(TOKEN_COUNT, width)
        self.metadata = nn.ModuleList(nn.Embedding(size, width) for size in vocab_sizes)
        self.clock = nn.Linear(1, width)
        self.dilations = (1, 2, 4, 8, 16)
        self.history = nn.ModuleList(nn.Conv1d(width, width, 3, dilation=d) for d in self.dilations)
        self.out = nn.Linear(width, TOKEN_COUNT)

    def forward(self, keys, metadata, position):
        h = self.keys(keys) + self.clock(position.unsqueeze(-1))
        for index, embedding in enumerate(self.metadata):
            h = h + embedding(metadata[:, index]).unsqueeze(1)
        h = h.transpose(1, 2)
        for dilation, layer in zip(self.dilations, self.history):
            h = h + .25 * F.silu(layer(F.pad(h, (2 * dilation, 0))))
        return self.out(h.transpose(1, 2))


@torch.inference_mode()
def assess(model, stream, group, args, sequence):
    model.eval()
    counts = Counter()
    for keys, targets, metadata, position in batches(
            stream, group, np.arange(len(group[0])), args.device, sequence, args.batch_size):
        prediction = model(keys, metadata, position).reshape(*targets.shape, 8, 9).logsumexp(-2).argmax(-1)
        earlier, actual = keys % 9, targets % 9
        changed = earlier != actual
        correct = prediction == actual
        counts['decisions'] += targets.numel()
        counts['right_direction'] += int(correct.sum())
        counts['previous_direction_matches'] += int((~changed).sum())
        counts['direction_changes'] += int(changed.sum())
        counts['right_after_change'] += int((changed & correct).sum())
    return dict(counts, direction_accuracy=counts['right_direction'] / counts['decisions'],
                keep_last_accuracy=counts['previous_direction_matches'] / counts['decisions'],
                direction_change_accuracy=counts['right_after_change'] / max(1, counts['direction_changes']))


def train(data, out, args):
    torch.set_num_threads(1)
    torch.manual_seed(508)
    torch.set_float32_matmul_precision('high')
    stream = np.load(data / 'tokens.npy', mmap_mode='r')
    spans = json.loads((data / 'stages.json').read_text(encoding='utf-8'))
    vocab = {key: {value: i for i, value in enumerate(
        sorted({stage[key] for stage in spans if stage['split'] == 'train'}), 1)}
        for key in VOCAB_FIELDS}
    groups = {name: tile(spans, name, args.sequence, vocab) for name in ('train', 'validation', 'test')}
    sizes = [len(vocab[key]) + 1 for key in VOCAB_FIELDS]
    model = ActionPrior(sizes).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    rng = np.random.default_rng(508)
    group = groups['train']
    seen = set()
    report = {'sequence_decisions': args.sequence, 'epochs': [], 'training_tiles_available': len(group[0])}
    out.mkdir(parents=True, exist_ok=False)
    best = -1.
    for epoch in range(args.epochs):
        order = rng.permutation(len(group[0]))
        if args.train_tokens:
            order = order[:max(1, args.train_tokens // args.sequence)]
        seen.update(order.tolist())
        model.train()
        total_loss = torch.zeros((), device=args.device)
        for number, (keys, targets, metadata, position) in enumerate(
                batches(stream, group, order, args.device, args.sequence, args.batch_size), 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16,
                                enabled=torch.device(args.device).type == 'cuda'):
                logits = model(keys, metadata, position)
                loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            loss.backward()
            optimizer.step()
            total_loss += loss.detach()
            if number % 100 == 0:
                print(f'epoch {epoch + 1}, {min(number * args.batch_size, len(order)) * args.sequence} tokens',
                      flush=True)
        row = {'epoch': epoch + 1, 'trained_decisions': len(order) * args.sequence,
               'mean_batch_nll': float(total_loss / number),
               'validation': assess(model, stream, groups['validation'], args, args.sequence)}
        report['epochs'].append(row)
        with (out / 'epochs.jsonl').open('a', encoding='utf-8') as stream_log:
            stream_log.write(json.dumps(row) + '\n')
        print(row, flush=True)
        score = row['validation']['direction_change_accuracy']
        if score > best:
            best = score
            report['selected_epoch'] = epoch + 1
            torch.save(model.state_dict(), out / 'prior.pt')
    model.load_state_dict(torch.load(out / 'prior.pt', weights_only=True, map_location=args.device))
    report['unique_train_decisions'] = len(seen) * args.sequence
    report['test'] = assess(model, stream, groups['test'], args, args.sequence)
    (out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    (out / 'vocab.json').write_text(json.dumps(vocab, indent=2) + '\n', encoding='utf-8')
    print(report['test'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pack native action streams and train a causal action-only prior.')
    commands = parser.add_subparsers(dest='command', required=True)
    packing = commands.add_parser('pack')
    packing.add_argument('--corpus', type=Path, required=True)
    packing.add_argument('--out', type=Path, required=True)
    training = commands.add_parser('train')
    training.add_argument('--data', type=Path, required=True)
    training.add_argument('--out', type=Path, required=True)
    training.add_argument('--epochs', type=int, default=1)
    training.add_argument('--train-tokens', type=int, default=0)
    training.add_argument('--sequence', type=int, default=64)
    training.add_argument('--batch-size', type=int, default=512)
    training.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.command == 'pack':
        pack(args.corpus, args.out)
    else:
        if min(args.epochs, args.sequence, args.batch_size) < 1 or args.train_tokens < 0:
            parser.error('Lengths and counts must be positive. Zero training tokens selects all available tiles.')
        train(args.data, args.out, args)
