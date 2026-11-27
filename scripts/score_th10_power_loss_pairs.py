"""Measure the trained Power-loss target on matched held TH10 original credits."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from scripts.collect_th10_full_action import save
from scripts.price_th10_unspent_power import death_costs
from scripts.run_th10_camera_cohort import digest


def score(directory):
    registered=json.loads((directory.parent/'camera-protocol.json').read_bytes())
    seeds=registered['eval_seeds']['10']
    arms=[registered['reference_arm'],registered['candidate']]
    rows={};groups={}
    for name in arms:
        credits=[]
        for seed in seeds:
            folder=directory/f'held-{name}'/f'accelerator-{seed}'
            r=json.loads((folder/'report.json').read_bytes())
            assert r['seed']==seed and r['arm']=='held-'+name and r['terminal'] and r['stage_sensor_validated']
            trace=folder/'native.csv.gz'
            assert digest(trace)==r['stage2_evidence']['native_trace_sha256']
            with np.load(folder/'behavior.npz') as z:
                cost,events=death_costs(trace,z['action_utc_ms'],z['reward_life'])
            with gzip.open(trace,'rt') as f:states=list(csv.DictReader(f))
            t=np.array([int(s['utc_ms']) for s in states]);life=np.array([int(s['lives']) for s in states])
            power=np.array([int(s['power_units']) for s in states]);dt=np.diff(t)
            valid=(dt<=80)&(life[1:]==life[:-1])&(life[:-1]>=0)
            positive_gain=float(np.maximum(np.diff(power)[valid],0).sum()/20.)
            confirmed=sum(e['charged_casts'] for e in events)
            assert abs(float(cost.sum())+.03*confirmed)<1e-5
            credits.append(dict(seed=seed,host=r['host'],policy_sha256=r['policy_sha256'],native_trace_sha256=digest(trace),
                original_casts=len(r['successful_bombs']),confirmed_payable_casts_lost=confirmed,
                observed_positive_same_life_power=positive_gain,measured_playable_s=float(dt[valid].sum()/1000),
                measured_time_with_payable_strike_s=float(dt[valid&(power[:-1]>=20)].sum()/1000),
                confirmed_in_final_death=sum(e['charged_casts'] for e in events if e['terminal']),
                ambiguous_native_x_edges=sum(e['nearby_native_x_edge'] for e in events),events=events))
        rows[name]=credits
        groups[name]={key:sum(r[key] for r in credits) for key in
            ('original_casts','confirmed_payable_casts_lost','confirmed_in_final_death','ambiguous_native_x_edges',
             'observed_positive_same_life_power','measured_playable_s','measured_time_with_payable_strike_s')}
        groups[name]['fraction_of_measured_time_with_payable_strike']=(
            groups[name]['measured_time_with_payable_strike_s']/groups[name]['measured_playable_s'])
    draws=np.random.default_rng(seeds[0]).integers(len(seeds),size=(20000,len(seeds)))
    comparisons={}
    for metric in ('original_casts','confirmed_payable_casts_lost','confirmed_in_final_death','observed_positive_same_life_power'):
        diff=np.array([r[metric]-l[metric] for l,r in zip(rows[arms[0]],rows[arms[1]])])
        comparisons[metric]=dict(mean_change=diff.mean(),paired_bootstrap95=np.quantile(diff[draws].mean(1),[.025,.975]).tolist())
    save(directory/'held-power-loss-events.json',dict(
        method='Held trajectories are rescored with the exact frozen training function death_costs(), using the 250 ms window. Measurement only. Counts of lost whole payable Spirit Strikes exclude ambiguous native X edges and inadequate observation windows. Positive Power is the sum of observed same-life gains across samples no more than 80 ms apart, including recovered drops; it is not an item count.',
        groups=groups,comparisons=comparisons,credits=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('native10',type=Path)
    score(p.parse_args().native10)
