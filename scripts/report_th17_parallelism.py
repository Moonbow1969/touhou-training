"""Summarize actual simultaneous native TH17 progress on the shared Gamer monotonic clock."""
from pathlib import Path
import argparse,json
from collections import defaultdict
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/rollout-worker-probe/th17-full'))
from timeline import progressed_ticks


def observed_activity(credits):
    """Partition the single host clock at every enrolled episode's first and last action."""
    boundaries=sorted({float(t) for _,ts,_ in credits for t in (ts[0],ts[-1])})
    seconds=defaultdict(float)
    bins=defaultdict(lambda:dict(decisions=0,interpolated_native_ticks=0.0))
    segments=[]
    for begin,end in zip(boundaries,boundaries[1:]):
        active=[(r,ts,ticks) for r,ts,ticks in credits if ts[0]<=begin and end<=ts[-1]]
        lanes=sorted(r['lane'] for r,_,_ in active)
        if len(set(lanes))!=len(lanes):
            raise ValueError('Concurrent original games share a physical lane')
        count=len(active)
        seconds[count]+=end-begin
        decisions=sum(int(ts.searchsorted(end,side='right')-ts.searchsorted(begin,side='right')) for _,ts,_ in active)
        ticks=sum(float(np.interp(end,ts,t)-np.interp(begin,ts,t)) for _,ts,t in active)
        bins[count]['decisions']+=decisions
        bins[count]['interpolated_native_ticks']+=ticks
        segments.append(dict(begin_monotonic_seconds=begin,end_monotonic_seconds=end,
                             active_lanes=lanes,decisions=decisions,interpolated_native_ticks=ticks))
    span=boundaries[-1]-boundaries[0]
    union=sum(dt for n,dt in seconds.items() if n>0)
    actions=sum(len(ts)-1 for _,ts,_ in credits)
    native_ticks=sum(int(t[-1]-t[0]) for _,_,t in credits)
    if (not np.isclose(sum(seconds.values()),span) or sum(v['decisions'] for v in bins.values())!=actions
            or not np.isclose(sum(v['interpolated_native_ticks'] for v in bins.values()),native_ticks)):
        raise ValueError('Original action interval union fails to reconcile to recorded clocks')
    def rates(duration):
        return dict(seconds=duration,decisions_per_second=actions/duration,native_ticks_per_second=native_ticks/duration,
                    full_credits_per_hour=len(credits)*3600/duration)
    return dict(credits=len(credits),decisions=actions,observed_native_ticks=native_ticks,
                entire_cohort_span=rates(span),at_least_one_enrolled_game_active=rates(union),
                by_active_enrolled_lane_count={n:dict(seconds=dt,**bins[n]) for n,dt in sorted(seconds.items())},
                segments=segments,scope='One physical host clock, first policy onset through last observed action. The full span includes cold reset, packing, idle gaps and other studies occupying released lanes. The union excludes periods with zero enrolled games active. Counts describe only this cohort. Other workloads and fleet saturation are unmeasured. Per-segment ticks are linearly interpolated across observed native action boundaries; whole-cohort totals are exact.')

def report(root,episode_prefix=''):
    credits=[]
    for p in sorted(root.glob('*-*-*/receipt.json')):
        if episode_prefix and not p.parent.name.startswith(episode_prefix):continue
        r=json.loads(p.read_bytes())
        t=np.genfromtxt(p.with_name('decisions.tsv'),names=True,delimiter='\t')
        credits.append((r,np.r_[t['onset_ns'][0],t['observed_ns']]/1e9,progressed_ticks(np.r_[t['stage'][0],t['after_stage']],np.r_[t['tick'][0],t['after_tick']])))
    cell={};wall={}
    if not credits:return {}
    # A released instance can be collecting another study. Restrict the load
    # comparison to the common interval in which every observed lane belongs
    # to this cohort, so an uncounted successor cannot contaminate a 1/2-lane cell.
    lanes={r['lane'] for r,_,_ in credits}
    common_begin=max(min(ts[0] for r,ts,_ in credits if r['lane']==lane) for lane in lanes)
    common_end=min(max(ts[-1] for r,ts,_ in credits if r['lane']==lane) for lane in lanes)
    for start in np.arange(np.ceil(common_begin),np.floor(common_end)):
        spans=[(r,ts,ticks) for r,ts,ticks in credits if ts[0]<=start<start+1<=ts[-1]]
        if not spans:continue
        # Drop windows where a new episode enters partway or another exits early.
        if any(start<ts[0]<start+1 or start<ts[-1]<start+1 for _,ts,_ in credits):continue
        count=len(spans)
        keys={r['lane'] for r,_,_ in spans}
        if len(keys)!=count:raise ValueError('Two original credits appear concurrent in one lane')
        cell.setdefault(count,[]).append(sum(np.interp(start+1,ts,ticks)-np.interp(start,ts,ticks) for _,ts,ticks in spans))
        wall.setdefault(count,[]).append(sum(np.searchsorted(ts,start+1)-np.searchsorted(ts,start) for _,ts,ticks in spans))
    groups={}
    for count,values in cell.items():
        groups[count]=dict(overlap_seconds=len(values),mean_aggregate_native_ticks_per_second=float(np.mean(values)),
            median_aggregate_native_ticks_per_second=float(np.median(values)),p10_p90_aggregate_native_ticks_per_second=np.percentile(values,[10,90]).tolist(),
            mean_aggregate_four_head_decisions_per_second=float(np.mean(wall[count])))
    span=common_end-common_begin
    accounting=None
    if span>0:
        busy=sum(max(0,min(ts[-1],common_end)-max(ts[0],common_begin)) for _,ts,_ in credits)
        steps=sum(np.count_nonzero((ts[1:]>common_begin)&(ts[1:]<=common_end)) for _,ts,_ in credits)
        terminals=sum(common_begin<ts[-1]<=common_end for _,ts,_ in credits)
        gaps=[]
        for lane in lanes:
            runs=sorted((ts[0],ts[-1]) for r,ts,_ in credits if r['lane']==lane)
            gaps.extend(b[0]-a[-1] for a,b in zip(runs,runs[1:]) if common_begin<=a[-1]<=b[0]<=common_end)
        accounting=dict(physical_wall_seconds=float(span),decisions=int(steps),native_credit_terminals=int(terminals),
                        amortized_decisions_per_second=float(steps/span),native_terminals_per_hour=float(terminals*3600/span),
                        fraction_available_game_slots_acting=float(busy/(span*len(lanes))),
                        intercredit_gap_seconds_median=float(np.median(gaps)) if gaps else None,
                        intercredit_gap_seconds_p90=float(np.percentile(gaps,90)) if gaps else None,
                        gap_definition='Previous last action to next first action on same instance: includes title reset, UI evidence, packing, model readiness and dispatch.')
    return dict(credits=len(credits),original_cpu_quota_per_lane=2,host='Gamer-WSL',
                common_allocated_lanes=sorted(lanes),common_allocation_monotonic_seconds=[common_begin,common_end] if span>0 else None,
                scope='The common window intersects each enrolled lane’s first-to-last action envelope. It may be empty. Groups summarize complete one-second windows in that intersection, counting only this study. End-to-end rates for the full cohort and its observed active union are available separately.',
                groups=groups,end_to_end=accounting,observed_cohort_activity=observed_activity(credits))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    o=report(a.root);a.out.write_text(json.dumps(o,indent=2)+'\n');print(json.dumps(o,indent=2))

