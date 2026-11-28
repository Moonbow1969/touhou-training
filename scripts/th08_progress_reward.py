"""Deterministic reward for an evidence-qualified, single TH08 v0.02 credit.

All clocks are *native playable stage ticks*. Supply T+1 observed states for
T actual policy actions. The final tick of a Game Over/ending is the last
confirmed playable tick, even when the terminal UI has since reset its clock.
An unqualified clock or missing boundary is an error, never a zero-fill.
The caller is responsible for verifying the original image/build and events.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Integral

import numpy as np


GAMMA = .999


@dataclass(frozen=True)
class RewardTrace:
    progress: np.ndarray
    stage_bonus: np.ndarray
    potential: np.ndarray
    shaping: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
    bootstrap: float


def _ints(name, data, n=None):
    a = np.asarray(data)
    if a.ndim != 1 or a.dtype.kind not in "iu" or (n is not None and len(a) != n):
        raise ValueError(f"{name} requires one integer per observation or action")
    if a.dtype.kind == "u" and a.size and a.max().item() > np.iinfo(np.int64).max:
        raise ValueError(f"{name} exceeds int64")
    return a.astype(np.int64, copy=False)


def _integer(value):
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def progress_reward(*, stage, tick, spares, state_kind, continues, spare_delta,
                    stage_ticks: Mapping, boundaries: Mapping, life_weight,
                    base_bootstrap=None):
    """Compute the two registered arms with life_weight=0 or 1 and gamma=.999.

    state_kind is 'play' up to the verified endpoint. Its final label is
    'game_over', 'objective_terminal' (true task ending), or 'play' (truncation).
    spares are *reserves only*. The Game Over UI may display 0 or native -1.
    spare_delta[T] independently records earned/spent spare units, excluding
    loss of the last active life. Counts of continues must stay zero throughout.

    stage_ticks maps stage -> preregistered positive normalization length.
    boundaries maps an action index that crossed a stage to a verified pair:
    (last playable old-stage tick, first playable new-stage tick). The two
    same-stage intervals are each divided by their own stage length. The
    seam counts once via the stage bonus. Loading contributes no elapsed ticks.

    For a truncation, pass V_base(next) (including an explicit zero if justified).
    Use the returned shifted bootstrap with the shaped rewards in GAE. For any
    interior critic comparison the same transformation is V_shaped=V_base-Phi.
    """
    if not _integer(life_weight) or life_weight not in (0, 1):
        raise ValueError("life_weight must be the registered integer 0 or 1")
    st = _ints("stage", stage)
    if len(st) < 2 or np.any((st < 1) | (st > 3)):
        raise ValueError("Expected a nonempty trajectory in official trial stages 1–3")
    n = len(st) - 1
    ticks = _ints("tick", tick, n + 1)
    lives = _ints("spares", spares, n + 1)
    cont = _ints("continues", continues, n + 1)
    delta = _ints("spare_delta", spare_delta, n)
    kinds = np.asarray(state_kind)
    if kinds.shape != st.shape or not np.all(kinds[:-1] == "play"):
        raise ValueError("Only the final credited action can lead to a nonplay state")
    end = kinds[-1]
    if end not in ("play", "game_over", "objective_terminal"):
        raise ValueError("Missing verified game endpoint or playable truncation")
    if np.any(ticks < 0) or np.any(cont != 0):
        raise ValueError("Native playable ticks and a single no-Continue credit are required")
    if np.any(lives[:-1] < 0) or (end != "game_over" and lives[-1] < 0):
        raise ValueError("Negative playable reserve stock")
    if end == "game_over" and (lives[-2] != 0 or lives[-1] not in (-1, 0)):
        raise ValueError("Game Over requires an exhausted final active life")
    # A -1 native sentinel represents the active life, outside the spare ledger.
    if not np.array_equal(np.diff(np.maximum(lives, 0)), delta):
        raise ValueError("Observed spare changes disagree with the confirmed spare-event ledger")
    advance = np.diff(st)
    if np.any((advance < 0) | (advance > 1)):
        raise ValueError("Stage reset, skip, or second credit inside trajectory")
    seams = np.flatnonzero(advance)
    if (not all(_integer(i) for i in boundaries)
            or set(boundaries) != set(seams)):
        raise ValueError("Each stage transition requires exactly one verified boundary")
    if end == "game_over" and advance[-1]:
        raise ValueError("Game Over and a stage increment on the same edge are ambiguous")
    for s in np.unique(st):
        length = stage_ticks.get(int(s))
        if not _integer(length) or length <= 0:
            raise ValueError(f"Missing positive preregistered T_stage for stage {s}")
    denom = np.array([stage_ticks[int(s)] for s in st], dtype=np.float64)
    dt = np.diff(ticks)
    if np.any(dt[advance == 0] < 0):
        raise ValueError("Nonmonotonic tick within the same playable stage")
    progress = dt.astype(np.float64) / denom[:-1]
    for i in seams:
        pair = boundaries[i]
        if len(pair) != 2 or not all(_integer(t) for t in pair):
            raise ValueError("Boundary requires two observed integer ticks")
        old_last, new_first = (int(t) for t in pair)
        if old_last < ticks[i] or not 0 <= new_first <= ticks[i + 1]:
            raise ValueError("Boundary fails to bracket the two playable observations")
        progress[i] = ((old_last - ticks[i]) / denom[i]
                       + (ticks[i + 1] - new_first) / denom[i + 1])

    phi = life_weight * (lives.astype(np.float64) + 1.)
    done = np.zeros(n, dtype=bool)
    if end == "play":
        if base_bootstrap is None or not isfinite(base_bootstrap):
            raise ValueError("Playable truncation requires a finite unshaped critic bootstrap")
        bootstrap = float(base_bootstrap) - phi[-1]
    else:
        if base_bootstrap is not None and base_bootstrap != 0:
            raise ValueError("A true terminal has zero bootstrap")
        phi[-1] = 0.
        done[-1] = True
        bootstrap = 0.
    shaping = GAMMA * phi[1:] - phi[:-1]
    bonus = advance.astype(np.float64)
    return RewardTrace(progress, bonus, phi, shaping, progress + bonus + shaping, done, bootstrap)
