import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from scripts.pretrain_policy import CONTEXT_SIZE, LEVELS, condition_context


HORIZONS = (4, 12, 24)
GRID = 24
CELL = 8
BUTTONS = np.array([1, 2, 8, 16, 32, 64, 128], dtype=np.uint16)


def occupancy(bullets, position):
    row, column = np.floor(position / CELL + GRID / 2).astype(np.int64).transpose(2, 0, 1)[::-1]
    present = (bullets[..., 6] > 0) & (bullets[..., 5] > 0)
    inside = present & (row >= 0) & (row < GRID) & (column >= 0) & (column < GRID)
    batch, entity = np.nonzero(inside)
    output = np.zeros((len(bullets), GRID, GRID), dtype=np.float32)
    output[batch, row[batch, entity], column[batch, entity]] = 1
    return output


def examples(shards):
    for meta_path, meta in shards:
        with np.load(meta_path.with_suffix('.npz')) as source:
            frames, bullets, context = (source[key] for key in ('stage_frame', 'bullets', 'context'))
        action_path = meta_path.parent.parent / f"{Path(meta['name']).stem}-actions.npz"
        with np.load(action_path) as source:
            words = source[f"stage_{meta['stage']}_input_words"]
        position = bullets[..., :2] * (384, 448)
        player = context[..., :2] * (192, 448)
        context = condition_context(context, LEVELS.index(meta['difficulty']), meta['shot'])
        for horizon in HORIZONS:
            ends = np.searchsorted(frames, frames + horizon).clip(0, len(frames) - 1)
            replay_starts = frames + meta['offset']
            exact = (frames[ends] == frames + horizon) & (replay_starts + horizon < len(words))
            start = np.flatnonzero(exact)
            if not len(start):
                continue
            end = ends[start]
            displacement = player[end] - player[start]
            current = occupancy(bullets[start], position[start])
            projected = position[start] + bullets[start, :, 2:4] * 16 * horizon
            ballistic = occupancy(bullets[start], projected)
            target = occupancy(bullets[end], position[end] + displacement[:, None, :])
            action_window = words[replay_starts[start, None] + np.arange(1, horizon + 1)]
            actions = np.zeros((len(start), max(HORIZONS), len(BUTTONS)), dtype=np.float32)
            actions[:, :horizon] = (action_window[..., None] & BUTTONS) != 0
            elapsed = np.full((len(start), 1), horizon / max(HORIZONS), dtype=np.float32)
            condition = np.concatenate((context[start], elapsed, actions.reshape(len(start), -1)), axis=1)
            features = np.stack((current, ballistic), axis=1)
            values = (features, condition, target, (displacement / 32).astype(np.float32))
            yield meta, horizon, TensorDataset(*(torch.from_numpy(value) for value in values))


class WorldModel(nn.Module):
    def __init__(self, conditions, width=32):
        super().__init__()
        self.condition = nn.Sequential(nn.Linear(conditions, 128), nn.SiLU(), nn.Linear(128, width))
        self.encoder = nn.Conv2d(2, width, 3, padding=1)
        self.decoder = nn.Sequential(nn.SiLU(), nn.Conv2d(width, width, 5, padding=2), nn.SiLU(),
                                     nn.Conv2d(width, 1, 3, padding=1))
        self.player = nn.Linear(width, 2)

    def forward(self, grid, condition):
        features = self.condition(condition)
        residual = self.decoder(self.encoder(grid) + features[:, :, None, None]).squeeze(1)
        return residual + 4 * grid[:, 1] - 2, self.player(features)


def evaluate(model, data, device, near_player_cells=0):
    predictions, offsets = [], []
    with torch.inference_mode():
        for grid, condition, _, _ in DataLoader(data, batch_size=128):
            pred, displacement = model(grid.to(device), condition.to(device))
            predictions.append(pred.cpu())
            offsets.append(displacement.cpu())
    grids, _, target, displacement = data.tensors
    truth = target.numpy().ravel()
    forecasts = (('persistence', grids[:, 0].numpy()), ('constant_velocity', grids[:, 1].numpy()),
                 ('learned', torch.cat(predictions).numpy()))
    scores = {name: round(average_precision(truth, pred.ravel()), 4)
              for name, pred in forecasts} if truth.any() else {}
    report = {
        'pairs': len(data), 'occupied_cell_fraction': float(truth.mean()), 'occupancy_ap': scores,
        'player_mae_game_units': round(float((torch.cat(offsets) - displacement).abs().mean() * 32), 3),
        'stationary_player_mae_game_units': round(float(displacement.abs().mean() * 32), 3),
    }
    if near_player_cells:
        cells = np.floor(displacement.numpy() * (32 / CELL) + GRID / 2)
        in_view = np.all((cells >= 0) & (cells < GRID), axis=1)
        indices = np.arange(GRID)
        x_near = np.abs(indices[None, :] - cells[:, 0:1]) <= near_player_cells
        y_near = np.abs(indices[None, :] - cells[:, 1:2]) <= near_player_cells
        region = in_view[:, None, None] & x_near[:, None] & y_near[:, :, None]
        local_truth = target.numpy()[region]
        report['near_player_pairs'] = int(in_view.sum())
        report['near_player_positive_cells'] = int(local_truth.sum())
        report['near_player_occupancy_ap'] = {
            name: round(average_precision(local_truth, pred[region]), 4) for name, pred in forecasts
        } if local_truth.any() else {}
    return report


def average_precision(truth, scores):
    order = np.argsort(scores, kind='stable')[::-1]
    sorted_scores = scores[order]
    threshold_ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(order) - 1]
    true_positives = np.cumsum(truth[order], dtype=np.int64)[threshold_ends]
    precision = true_positives / (threshold_ends + 1)
    return float(precision @ np.diff(true_positives, prepend=0) / true_positives[-1])


def main():
    parser = argparse.ArgumentParser(description='Forecast local projectile occupancy and player motion.')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--shards-per-block', type=int, default=12)
    args = parser.parse_args()
    if min(args.shards_per_block, args.epochs, args.width) < 1:
        parser.error('Epochs, width and shards per block must be positive.')
    torch.manual_seed(79)
    torch.set_num_threads(1)
    shards = [(path, json.loads(path.read_text(encoding='utf-8'))) for path in sorted(args.data.glob('*.json'))]
    train = [(path, meta) for path, meta in shards if meta['split'] == 'train']
    val = [(path, meta) for path, meta in shards if meta['split'] == 'validation']
    if not train or not val:
        raise ValueError('Training and validation each need aligned replay stages with exact frame gaps.')
    player_groups = [{meta.get('player_group', meta['player'].casefold()) for _, meta in split}
                     for split in (train, val)]
    if player_groups[0] & player_groups[1]:
        raise ValueError('A replay player occurs in both training and validation.')
    conditions = CONTEXT_SIZE + 1 + max(HORIZONS) * len(BUTTONS)
    model = WorldModel(conditions, args.width).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4)
    positive_weight = torch.tensor(8.0, device=args.device)
    rng = np.random.default_rng(79)
    args.out.mkdir(parents=True, exist_ok=False)
    history = []
    for epoch in range(args.epochs):
        seen = 0
        loss_total = torch.zeros((), device=args.device)
        order = rng.permutation(len(train)) if len(train) > args.shards_per_block else range(len(train))
        for start in range(0, len(order), args.shards_per_block):
            block = [train[index] for index in order[start:start + args.shards_per_block]]
            groups = [data for _, _, data in examples(block)]
            if not groups:
                continue
            train_data = ConcatDataset(groups)
            seen += len(train_data)
            for batch in DataLoader(train_data, batch_size=128, shuffle=True):
                grid, condition, occupancy_target, displacement = (x.to(args.device) for x in batch)
                pred, offset = model(grid, condition)
                loss = F.binary_cross_entropy_with_logits(pred, occupancy_target, pos_weight=positive_weight)
                loss += F.smooth_l1_loss(offset, displacement)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_total += loss.detach() * len(batch[0])
        if not seen:
            raise ValueError('Training stages contain no exact frame transitions.')
        entry = {'epoch': epoch + 1, 'transitions': seen, 'train_loss': loss_total.item() / seen}
        history.append(entry)
        with (args.out / 'epochs.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(entry) + '\n')
        print(json.dumps(entry), flush=True)
    validation = [dict(replay=meta['name'], title=meta['title'], stage=meta['stage'], horizon_frames=horizon,
                       **evaluate(model, data, args.device)) for meta, horizon, data in examples(val)]
    if not validation:
        raise ValueError('Validation stages contain no exact frame transitions.')
    report = {
        'config': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'parameters': sum(p.numel() for p in model.parameters()), 'width': args.width,
        'epochs': args.epochs, 'training_pairs': seen,
        'training_titles': sorted({meta['title'] for _, meta in train}),
        'validation': validation, 'history': history,
    }
    torch.save(model.state_dict(), args.out / 'world.pt')
    (args.out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
