"""Verify the H1 wire byte against the immediately preceding accepted movement."""
import numpy as np


def previous_move_fields(decisions, actions):
    if 'previous_move' not in decisions.dtype.names:
        return {}
    observed = decisions['previous_move']
    if not np.array_equal(observed, np.r_[4, actions[:-1, 0]]):
        raise ValueError('The H1 byte differs from the previous original action')
    return {'previous_move': observed.astype(np.uint8)}
