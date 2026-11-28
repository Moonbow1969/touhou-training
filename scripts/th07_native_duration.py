"""Value targets for TH07 decisions with independently measured native durations."""

import numpy as np


def native_discount(ticks, gamma=.999):
    ticks = np.asarray(ticks)
    if ticks.ndim != 1 or not len(ticks) or ticks.dtype.kind not in "iu" or np.any(ticks < 1):
        raise ValueError("Each sampled decision requires a positive observed native duration")
    if not 0 < gamma <= 1:
        raise ValueError("The four-tick discount must be in (0,1]")
    return np.power(gamma, ticks.astype(np.float64) / 4)


def actor_eligible(requested, executed, ticks):
    """Use all native words of a transition to distinguish a hold from key leakage."""
    requested, executed, ticks = (np.asarray(x) for x in (requested, executed, ticks))
    native_discount(ticks)
    if (requested.shape != ticks.shape or executed.shape != (int(ticks.sum()),)
            or any(x.dtype.kind not in "iu" or np.any((x < 0) | (x > 255) | ((x & 8) != 0))
                   for x in (requested, executed))):
        raise ValueError("Missing or invalid complete native key slots")
    offsets = np.r_[0, np.cumsum(ticks[:-1], dtype=np.int64)]
    return np.logical_and.reduceat(executed == np.repeat(requested, ticks), offsets)


def shape_progress(base_reward, potential, discounts):
    """The caller supplies within-window discounted event rewards and endpoint potential."""
    base, potential, discounts = (np.asarray(x, dtype=np.float64) for x in (base_reward, potential, discounts))
    if (base.ndim != 1 or not len(base) or discounts.shape != base.shape or potential.shape != (len(base) + 1,)
            or not all(np.isfinite(x).all() for x in (base, potential, discounts))
            or np.any((discounts <= 0) | (discounts > 1))):
        raise ValueError("Invalid native reward, state potential or per-transition discount")
    return base + discounts * potential[1:] - potential[:-1]


def duration_gae(rewards, values, discounts, lam=.985, *, bootstrap=0.):
    """Apply an observed-duration discount and one trace factor per sampled decision."""
    rewards, values, discounts = (np.asarray(x, dtype=np.float64) for x in (rewards, values, discounts))
    if (rewards.ndim != 1 or not len(rewards) or values.shape != rewards.shape or discounts.shape != rewards.shape
            or not all(np.isfinite(x).all() for x in (rewards, values, discounts))
            or np.any((discounts <= 0) | (discounts > 1)) or not np.isfinite(bootstrap) or not 0 <= lam <= 1):
        raise ValueError("Expected finite original-state values and valid decision discounts")
    result = np.empty_like(rewards)
    following_value, advantage = float(bootstrap), 0.
    for index in range(len(rewards) - 1, -1, -1):
        discount = discounts[index]
        residual = rewards[index] + discount * following_value - values[index]
        advantage = residual + lam * discount * advantage
        following_value = values[index]
        result[index] = advantage
    return result
