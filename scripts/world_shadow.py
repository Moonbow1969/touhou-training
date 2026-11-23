"""Score a world model on real policy rollouts without changing actions."""

import json

import numpy as np
import torch

from scripts.pretrain_policy import CONTEXT_SIZE
from scripts.stage_policy import action_keys
from scripts.train_world import BUTTONS, CELL, GRID, HORIZONS, WorldModel, average_precision, occupancy


class WorldShadow:
    def __init__(self, checkpoint, device, stride):
        sidecar = checkpoint.with_name('metrics.json')
        width = json.loads(sidecar.read_text(encoding='utf-8'))['width']
        conditions = CONTEXT_SIZE + 1 + max(HORIZONS) * len(BUTTONS)
        self.model = WorldModel(conditions, width).to(device).eval()
        self.model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        self.device = device
        self.stride = stride
        self.reset()

    def reset(self):
        self.samples = 0
        self.impacts = 0
        self.truth = []
        self.ballistic = []
        self.learned = []

    @torch.inference_mode()
    def observe(self, before, after, action, impact):
        horizon = 4
        old_bullets = before['bullets'][None]
        position = old_bullets[..., :2] * (384, 448)
        current = occupancy(old_bullets, position)
        projected = position + old_bullets[..., 2:4] * (16 * horizon)
        ballistic = occupancy(old_bullets, projected)
        displacement = (after['context'][:2] - before['context'][:2]) * (192, 448)
        future_bullets = after['bullets'][None]
        target = occupancy(future_bullets, future_bullets[..., :2] * (384, 448) + displacement)
        center = np.floor(displacement / CELL + GRID / 2).astype(int)
        if ((center < 0) | (center >= GRID)).any():
            return
        axes = np.arange(GRID)
        region = (np.abs(axes[:, None] - center[1]) <= 3) & (np.abs(axes[None, :] - center[0]) <= 3)
        keys = action_keys(action)
        pressed = [key in keys for key in ('z', 'x', 'shift', 'up', 'down', 'left', 'right')]
        actions = np.zeros((max(HORIZONS), len(BUTTONS)), dtype=np.float32)
        actions[:horizon] = pressed
        condition = np.concatenate((before['context'][:CONTEXT_SIZE],
                                    np.array([horizon / max(HORIZONS)], dtype=np.float32), actions.ravel()))
        grid = torch.from_numpy(np.stack((current, ballistic), axis=1)).to(self.device)
        pred, _ = self.model(grid, torch.from_numpy(condition[None]).to(self.device))
        self.samples += 1
        self.impacts += int(impact)
        self.truth.append(target[0, region])
        self.ballistic.append(ballistic[0, region])
        self.learned.append(pred[0].cpu().numpy()[region])

    def summary(self):
        row = {'horizon_frames': 4, 'samples': self.samples, 'impact_samples': self.impacts}
        if self.truth:
            truth = np.concatenate(self.truth)
            row['occupied_cells'] = int(truth.sum())
            if truth.any():
                row['ballistic_ap'] = average_precision(truth, np.concatenate(self.ballistic))
                row['learned_ap'] = average_precision(truth, np.concatenate(self.learned))
        return row
