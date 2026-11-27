"""Rescore original stage clock and consumed key delivery from a complete policy credit."""
import hashlib
from pathlib import Path
import numpy as np

def evaluate(raw,clock):
    def read(name):
        return np.genfromtxt(raw/name,names=True,delimiter='\t',dtype=None,encoding='utf8',ndmin=1)
    decisions=read('engineering-decisions.tsv');state=read('engineering-native.tsv');control=read('decisions.tsv')
    n=len(decisions)
    if len(control)!=n or not np.array_equal(decisions['action'],np.arange(n)) or not np.array_equal(decisions['observed_x11'],control['keys']) or not np.array_equal(decisions['observed_x11'],decisions['requested_x11']):raise ValueError('Physical key evidence or decision count differs')
    same=decisions['pre_stage'][1:]==decisions['pre_stage'][:-1]
    gaps=np.diff(decisions['pre_tick'])[same]
    if clock=='tick4' and (np.any(decisions['pre_tick']%4) or np.any(gaps<=0) or np.any(gaps%4)):raise ValueError('Invalid four-native-tick clock evidence')
    expected=sum(((decisions['requested_x11']>>bit)&1)*mask for bit,mask in enumerate((64,128,16,32,8,1,2)))
    observed=[];indices=[]
    for i,r in enumerate(decisions):
        if r['apply_stage']!=r['post_stage']:continue
        valid=(state['stage']==r['apply_stage']) & (state['stage']==state['manager_stage']) & (state['mode']==0) & (state['tick']>r['apply_tick']) & (state['tick']<=r['post_tick'])
        if valid.any():
            observed.append(int(state['keys'][np.flatnonzero(valid)[-1]]) & 251);indices.append(i)
    matches=int(np.sum(np.asarray(observed)==expected[indices]))
    if len(indices)<.90*n or matches<.95*len(indices):raise ValueError(('Original input verification below quality threshold',n,len(indices),matches))
    return dict(clock=clock,decisions=n,native_key_matches=matches,eligible_native_end_of_hold=len(indices),native_key_match_rate=matches/len(indices),physical_key_matches=n,
                native_gap_histogram={str(int(k)):int((gaps==k).sum()) for k in np.unique(gaps)},native_gap_mean=float(gaps.mean()),same_stage_gaps=len(gaps),
                native_trace_sha256=hashlib.sha256((raw/'engineering-native.tsv').read_bytes()).hexdigest(),
                decision_trace_sha256=hashlib.sha256((raw/'engineering-decisions.tsv').read_bytes()).hexdigest(),
                wall_mean_between_onsets_ms=float(np.diff(control['onset_ns']).mean()/1e6),camera_clock_crossings_or_retries=int(decisions['capture_retries'].sum()))
