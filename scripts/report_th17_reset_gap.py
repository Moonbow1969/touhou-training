"""Partition the original TH17 inter-credit gap using timestamps that actually exist."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np

def summary(values):
    return dict(n=len(values),mean_seconds=float(np.mean(values)),median_seconds=float(np.median(values)),
                p10_seconds=float(np.percentile(values,10)),p90_seconds=float(np.percentile(values,90)))

def report(root):
    by_lane=defaultdict(list)
    for p in root.glob('*-*-*/receipt.json'):
        r=json.loads(p.read_bytes());raw=p.parent
        t=np.genfromtxt(raw/'decisions.tsv',names=True,delimiter='\t')
        by_lane[r["lane"]].append((float(t['onset_ns'][0])/1e9,float(t['observed_ns'][-1])/1e9,raw))
    for key in by_lane:by_lane[key].sort()
    start=max(runs[0][0] for runs in by_lane.values())
    end=min(runs[-1][1] for runs in by_lane.values())
    pairs=[];excluded=[]
    for lane,runs in by_lane.items():
        for previous,next_credit in zip(runs,runs[1:]):
            if not start<=previous[1]<=next_credit[0]<=end:continue
            prev,after=previous[2],next_credit[2]
            gap=next_credit[0]-previous[1]
            times=[p.stat().st_mtime_ns/1e9 for p in (prev/'terminal.png',prev/'receipt.json',after/'rank-initial.png',after/'rank-selected.png')]
            segments=np.diff(times);remainder=gap-sum(segments)
            if min(segments)<0 or remainder<0:
                excluded.append(dict(previous=prev.name,following=after.name,gap_seconds=gap,host_file_deltas=segments.tolist(),reason='Original file timestamp order no longer supports decomposition'))
                continue
            pairs.append(dict(previous=prev.name,following=after.name,lane=lane,native_gap_seconds=gap,terminal_photo_to_sealed_seconds=segments[0],
                sealed_to_next_initial_rank_menu_seconds=segments[1],normal_rank_menu_to_selected_rank_seconds=segments[2],unseparated_outer_seconds=remainder))
    fields=('native_gap_seconds','terminal_photo_to_sealed_seconds','sealed_to_next_initial_rank_menu_seconds',
            'normal_rank_menu_to_selected_rank_seconds','unseparated_outer_seconds')
    return dict(scope='First-generation 3-lane common allocation. The total gap uses the shared Linux monotonic action timestamps. Differences of preserved host file timestamps partition inner phases. The residual joins final action→terminal photo and selected rank photo→next first action. It does not attribute either interval individually.',
                details={'terminal_photo_to_sealed_seconds':'Read original native TSV, pre-action frames, decisions, compress NPZ, hash and write the signed receipt.',
                         'sealed_to_next_initial_rank_menu_seconds':'Scheduler/model readiness, verified game executable, title reentry and difficulty-menu screenshot; older journals do not time each operation separately.',
                         'normal_rank_menu_to_selected_rank_seconds':'Navigate from photographed Normal rank to the selected original Hard or Lunatic rank and save photograph.',
                         'unseparated_outer_seconds':'Last applied action to previous terminal photo, plus next selected rank to character/beast selection, native observer/policy readiness and the next original first action.'},
                accepted=len(pairs),excluded=excluded,summary={k:summary([r[k] for r in pairs]) for k in fields},pairs=pairs)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    o=report(a.root);a.output.write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o['summary'],indent=2))
