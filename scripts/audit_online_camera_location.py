"""Check frozen player localization on agent-visited original TH10 held credits."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.probe_full_action_player_vision import read
from scripts.score_original_camera_run import records
from scripts.train_replay_camera_encoder import CameraPolicy, save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--weights', nargs='+', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads((args.root/'camera-protocol.json').read_bytes())
    verified = json.loads((args.root/'evaluation/credits.json').read_bytes())
    want = {(r['seed'], r['policy_sha256']): r for r in verified if r['title'] == 10}
    if len(want) != 2*len(protocol['eval_seeds']['10']):
        raise ValueError('Incomplete paired source')
    torch.set_num_threads(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = {}
    for p in args.weights:
        net = CameraPolicy().to(device).eval()
        net.load_state_dict(torch.load(p, weights_only=True, map_location=device))
        models[p.parent.name] = net
    credits = []
    seen = set()
    with torch.inference_mode():
        for _, path in records(args.root, True, True):
            if path.name != 'report.json':
                continue
            receipt = json.loads(path.read_bytes())
            key = receipt.get('seed'), receipt.get('policy_sha256')
            if key not in want:
                continue
            if key in seen or sha(path) != want[key]['native_log_sha256']:
                raise ValueError('Duplicate or changed held credit')
            seen.add(key)
            images, _, target, alignment_ms = read(path.parent)
            row = dict(seed=key[0], behavior=want[key]['arm'], verified_input_sha256=key[1],
                       receipt_sha256=sha(path), frames=len(target), max_alignment_ms=float(alignment_ms.max()), metrics={})
            for name, model in models.items():
                indices = []
                for batch in torch.from_numpy(images).split(64):
                    latent = model.spatial(batch.to(device).float()/255)
                    indices.append(model.player(latent).flatten(1).argmax(1).cpu().numpy())
                index = np.concatenate(indices)
                predicted = np.column_stack((index % 48, index // 48))*2
                errors = np.linalg.norm(4*(target-predicted), axis=1)
                row['metrics'][name] = dict(median_error_px=float(np.median(errors)),
                                            fraction_within_16_px=float(np.mean(errors <= 16)))
            credits.append(row)
    if seen != set(want):
        raise ValueError('Missing original source')
    summary = {name:dict(median_credit_median_px=float(np.median([r['metrics'][name]['median_error_px'] for r in credits])),
                         mean_credit_within_16_px=float(np.mean([r['metrics'][name]['fraction_within_16_px'] for r in credits]))) for name in models}
    save_json(args.out, dict(models={p.parent.name:sha(p) for p in args.weights}, summary=summary, credits=credits,
                             method='Same held original TH10 gameplay images for all frozen models. Native positions are diagnostic labels only. At most 40 ms between the original position and action capture; exclude 650 ms either side of a stock change. Credits each have equal weight.'))


if __name__ == '__main__':
    main()
