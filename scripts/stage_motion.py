"""Calibrate the player's speed from recent exact native moves."""

import numpy as np

from scripts.stage_risk import player_paths


class MotionCalibrator:
    def __init__(self):
        self.reset()
        self.slow_predictions = 0
        self.scale_updates = 0

    def reset(self):
        self.speed = 1.
        self.evidence_frame = None

    def predict(self, frame):
        if self.evidence_frame is not None and frame - self.evidence_frame > 40:
            self.reset()
        self.slow_predictions += self.speed < 1
        return self.speed

    def observe(self, before, after, action, frame):
        position_scale = np.array([192, 448], dtype=np.float32)
        position = after['context'][:2] * position_scale
        if np.linalg.norm(position - before['context'][:2] * position_scale) <= 1:
            return
        move, focus = map(int, action[:2])
        expected = [player_paths(before, 4, speed)[focus, move, -1] for speed in (1., 1 / 3)]
        normal_error, slow_error = np.linalg.norm(expected - position, axis=-1)
        if min(normal_error, slow_error) <= .5 and abs(normal_error - slow_error) > .75:
            speed = 1 / 3 if slow_error < normal_error else 1.
            self.scale_updates += self.speed != speed
            self.speed = speed
            self.evidence_frame = frame
