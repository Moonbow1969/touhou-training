import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch

from scripts.pretrain_policy import CONTEXT_SIZE
from scripts.train_world import BUTTONS, HORIZONS, WorldModel, evaluate, examples


def main():
    parser = argparse.ArgumentParser(description='Compare world models on identical native replay transitions.')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--split', choices=('validation', 'test'), default='test')
    parser.add_argument('--width', type=int, default=96)
    parser.add_argument('--near-player-cells', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.near_player_cells < 0:
        parser.error('Near-player radius must be nonnegative.')
    torch.set_num_threads(1)
    conditions = CONTEXT_SIZE + 1 + max(HORIZONS) * len(BUTTONS)
    models = {}
    for name, path in (('original', args.reference), ('corrected_data', args.candidate)):
        model = WorldModel(conditions, args.width).to(args.device).eval()
        model.load_state_dict(torch.load(path, map_location=args.device, weights_only=True))
        models[name] = model
    shards = []
    for path in sorted(args.data.glob('*.json')):
        meta = json.loads(path.read_text(encoding='utf-8'))
        if meta['split'] == args.split:
            shards.append((path, meta))
    rows = []
    for meta, horizon, data in examples(shards):
        rows.append({
            'replay': meta['name'], 'stage': meta['stage'], 'horizon': horizon,
            'models': {name: evaluate(model, data, args.device, args.near_player_cells)
                       for name, model in models.items()},
        })
    if not rows:
        raise ValueError(f'No exact native transitions in the {args.split} split.')
    totals = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for row in rows:
        for name, metrics in row['models'].items():
            values = totals[row['horizon']][name]
            for key in ('learned', 'constant_velocity'):
                if key in metrics['occupancy_ap']:
                    values[f'{key}_pairs'] += metrics['pairs']
                    values[key] += metrics['occupancy_ap'][key] * metrics['pairs']
                local = metrics.get('near_player_occupancy_ap', {})
                if key in local:
                    values[f'near_{key}_pairs'] += metrics['near_player_pairs']
                    values[f'near_{key}'] += local[key] * metrics['near_player_pairs']
            values['pairs'] += metrics['pairs']
            values['player_mae'] += metrics['player_mae_game_units'] * metrics['pairs']
    summary = {
        horizon: {
            name: {
                'pairs': int(values['pairs']),
                'player_mae_game_units': values['player_mae'] / values['pairs'],
                **{f'{key}_stage_ap_weighted_by_pairs': values[key] / values[f'{key}_pairs']
                   for key in ('learned', 'constant_velocity', 'near_learned', 'near_constant_velocity')
                   if values[f'{key}_pairs']},
            }
            for name, values in by_model.items()
        }
        for horizon, by_model in totals.items()
    }
    report = {'split': args.split, 'near_player_cells': args.near_player_cells, 'summary': summary, 'stages': rows}
    args.out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
