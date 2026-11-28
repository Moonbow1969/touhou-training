"""Causal, reset-per-credit camera and unsigned absolute difference pairs.

Stored originals require only the raw decision camera. A fully padded first
state repeats the first camera and its zero difference, including left padding.
"""
from collections import deque
import numpy as np


def frame_pair(frame, previous):
    delta = np.abs(frame.astype(np.int16) - previous).astype(np.uint8)
    return np.stack((frame, delta))


class CameraHistory:
    def __init__(self, states):
        if states not in (1, 4):
            raise ValueError("The original runner accepts one or four camera states")
        self.states = states
        self.history = deque(maxlen=states)
        self.previous = None

    def append(self, frame):
        if frame.shape != (112, 96) or frame.dtype != np.uint8:
            raise ValueError("Expected original uint8 playfield camera")
        first = self.previous is None
        pair = frame_pair(frame, frame if first else self.previous)
        if first:
            self.history.extend([pair] * (self.states - 1))
        self.history.append(pair)
        self.previous = frame
        return np.concatenate(self.history, axis=0)[None]


def reconstruct(frames, low, high, states):
    """Independently reconstruct an arbitrary slice within one original credit."""
    if frames.ndim != 3 or frames.shape[1:] != (112,96) or frames.dtype != np.uint8:
        raise ValueError("Invalid source frames")
    if states not in (1,4) or not 0 <= low < high <= len(frames):
        raise ValueError("Invalid original window bounds")
    indexes = np.maximum(np.arange(low, high)[:, None] + np.arange(1-states, 1), 0)
    gray = frames[indexes]
    older = frames[np.maximum(indexes - 1, 0)]
    delta = np.abs(gray.astype(np.int16) - older).astype(np.uint8)
    return np.stack((gray, delta), axis=2).reshape(high - low, 2 * states, 112, 96)
