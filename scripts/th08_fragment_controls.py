"""Verify consumed native inputs before admitting a restored TH08 fragment."""

import numpy as np

from scripts.audit_th08_live_tick4 import game_word
from scripts.run_th08_trial_camera import mask_for


def control_evidence(actions, slots, durations, terminal):
    size = len(actions)
    if size < 1 or slots.shape != (size,4) or durations.shape != (size,) or np.any(durations <= 0):
        raise ValueError("Fragment has incomplete native input windows")
    expected = np.asarray([game_word(mask_for(row)) for row in actions],dtype=np.int64)
    consumed = np.any(slots == expected[:,None],axis=1)
    windows = size - int(terminal)
    exact = (durations == 4) & np.all(slots == expected[:,None],axis=1)
    bad = np.flatnonzero(~exact[:windows])
    fraction = float(exact[:windows].mean()) if windows else 0.
    # The batch must still meet the global rate if a short fragment has one
    # independently identified first-slot delay after its startup check.
    short_partial = False
    if 0 < windows < 200 and len(bad) == 1:
        index = int(bad[0])
        if index >= 60 and durations[index] == 4 and expected[index-1] != expected[index]:
            requested = expected[index]
            short_partial = bool(np.array_equal(slots[index],[expected[index-1],requested,requested,requested]))
    first_exact = int(exact[:min(windows,60)].sum())
    qualified = (windows > 0 and consumed.all() and first_exact == min(windows,60)
                 and (fraction >= .995 or short_partial))
    return dict(windows=windows,exact=int(exact[:windows].sum()),first_exact=first_exact,
                consumed=int(consumed.sum()),short_partial=short_partial,fraction=fraction,qualified=bool(qualified))
