"""Admit a terminal input exclusion only when the original game's consumed keys prove it."""
import importlib.util
import hashlib
from pathlib import Path
import numpy as np


def playable_length(title,mask,reward,boundary):
    n=len(reward)
    if boundary is None and mask is None:return n
    if title!=12 or boundary is None or mask is None or n<2:
        raise ValueError('Unqualified post-terminal exclusion')
    if mask.dtype!=np.bool_ or mask.shape!=(n,) or mask[-1] or not mask[:-1].all():
        raise ValueError('Only a single trailing, unconsumed playable action may be excluded')
    if (boundary['raw_sampled_actions']!=n or boundary['live_actions']!=n-1 or
            boundary['first_excluded_action']!=n-1 or boundary['terminal_credit_action']!=n-2 or
            boundary['consumed_at_death_mask']==boundary['postdeath_consumed_mask'] or
            hashlib.sha256(mask.tobytes()).hexdigest()!=boundary['native_action_eligible_sha256'] or
            not boundary['consumed_terminal_action_start_tick']<=boundary['terminal_tick']<=boundary['consumed_terminal_action_end_tick'] or
            not boundary['pre_stock_record_ms']<boundary['terminal_stock_record_ms']<boundary['first_consumption_ms'] or
            float(reward[-1])!=0.):
        raise ValueError('Inconsistent original terminal attribution')
    return n-1


def verify_source(title,receipt,trace,folder):
    boundary=receipt.get('covered_native_events',{}).get('postterminal_input_boundary')
    if title!=12 and boundary is None:return None
    with np.load(trace,allow_pickle=False) as z:
        mask=z['native_action_eligible'] if 'native_action_eligible' in z else None
        if boundary is None and mask is None:return None
        reward=z['reward'].reshape(-1)
        playable_length(title,mask,reward,boundary)
        fields={k:z[k] for k in ('onset_ns','observed_ns')}
        parts=z['reward_parts']
    path=Path(__file__).resolve().parents[1]/'tools/rollout-worker-probe/th12-full/covered_events.py'
    spec=importlib.util.spec_from_file_location('th12_native_reward_evidence',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rebuilt=module.prepare(folder,fields,dict(receipt))
    if rebuilt!=receipt['covered_native_events'] or rebuilt['live_interval_gaps']:
        raise ValueError('Reconstructed TH12 original input consumption differs')
    np.testing.assert_array_equal(mask,fields['native_action_eligible'])
    np.testing.assert_array_equal(reward,fields['reward'].reshape(-1))
    np.testing.assert_array_equal(parts,fields['reward_parts'])
    return boundary
