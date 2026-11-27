"""Run original TH10 paired credits with qualified native stage telemetry."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import time
from types import SimpleNamespace

from scripts.collect_th10_full_action import save
from scripts.package_th10_credit_logs import package
from scripts.run_th10_camera_cohort import digest
from scripts.run_th10_camera_pair import run as paired


def run(root,workers,suite='normal'):
    root=root.resolve();workers=workers.resolve()
    calibration=root/'engineering/stage-positive-control/sensor-repair.json'
    proof=json.loads(calibration.read_bytes())
    assert proof['qualified'] and proof['corrected_register']=='0x472b40'
    experiments=[('v2-online','v2-supervised/joint-v2-3000',981051),
                 ('attention-online','spatial-policy-study/spatial-attention',1121051)]
    host='momentum';difficulty=1;arms=('held-sft','held-ppo')
    if suite=='easy':
        experiments=[('easy-curriculum','v2-online/policy',1101051)]
        host='accelerator';difficulty=0;arms=('held-v2-ppo','held-curriculum-ppo')
        while not (root/'easy-curriculum/native10/paired.json').is_file():time.sleep(5)
    if suite=='power-loss':
        experiments=[('power-loss','easy-curriculum/policy',1131051)]
        host='accelerator';difficulty=0;arms=('held-plain-curriculum','held-power-loss')
    for experiment,baseline,first in experiments:
        repeat=suite!='power-loss'
        new=root/(experiment+'-corrected-stage' if repeat else experiment)
        out=new/'native10';out.mkdir(parents=True,exist_ok=True)
        old=root/experiment/'native10'
        seeds=list(range(first,first+12));reference=root/baseline/'policy.ts';candidate=root/experiment/'policy/policy.ts'
        save(out/'correction-protocol.json',dict(source_campaign=experiment,original_controller_seeds=seeds,physical_host=host,difficulty=difficulty,
            fresh_original_credits=True,repeat_of_historical_controller_seeds=repeat,independent_native_game_rng=True,overwrites_historical_credit=False,
            corrected_stage_register=proof['corrected_register'],additional_negative_control_register=proof['legacy_register'],
            original_title_witness_sha256=digest(calibration),cadence_ms=120,
            actions='All four camera-policy heads, no geometry, aim intervention or automatic Bomb. Original full starting stock, no Continue.'))
        if suite=='power-loss':
            completed=root/'easy-curriculum-corrected-stage/th10-held-native-logs.zip'
            while not (completed.is_file() and candidate.is_file()):time.sleep(5)
        paired(SimpleNamespace(out=out,workers=workers,host=host,difficulty=difficulty,stage_sensor='qualified-v2',
            reference_name=arms[0],candidate_name=arms[1],reference=reference,candidate=candidate,
            seeds=seeds,concurrency=2 if difficulty==0 else 3,candidate_history=False))
        receipts=[]
        for arm in arms:
            for seed in seeds:
                folder=out/arm/f'{host}-{seed}';r=json.loads((folder/'report.json').read_bytes())
                with gzip.open(folder/'native.csv.gz','rt') as f:states=list(csv.DictReader(f))
                assert r['stage_sensor_validated'] and r['stage_register']=='0x472b40'
                assert r['terminal'] and not r['continues'] and {v['difficulty'] for v in states}=={str(difficulty)}
                assert int(states[0]['stage'])==1 and r['stage2_reached']==any(int(v['stage'])>=2 for v in states)
                older=old/arm/f'{host}-{seed}'/'report.json'
                if repeat and older.is_file():
                    original=json.loads(older.read_bytes())
                    assert r['seed']==original['seed'] and r['policy_sha256']==original['policy_sha256']
                    assert r['source']['bundle']!=original['source']['bundle'] and r['behavior_sha256']!=original['behavior_sha256']
                receipts.append(dict(arm=arm,seed=seed,host=r['host'],source_bundle=r['source']['bundle'],
                    native_action_executable_sha256=r['source']['linux_binary'],policy_sha256=r['policy_sha256'],
                    true_stage_register=r['stage_register'],native_stage_values=sorted({int(v['stage']) for v in states}),
                    legacy_values=sorted({int(v['legacy_start_field']) for v in states}),stage_entries=r['stage_entries'],
                    final_spares=r['final_lives'],final_power=r['final_power'],steps=r['steps'],native_rows=len(states),
                    matching_native_inputs=r['native_inputs_matching'],terminal=r['native_outcome'],raw_camera_sha256=digest(folder/'original-pixels.npz'),
                    native_trace_sha256=digest(folder/'native.csv.gz'),behavior_sha256=r['behavior_sha256']))
        save(out/'qualified-stage-receipts.json',receipts)
        package(new/'th10-held-native-logs.zip',[out/name for name in arms])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path);p.add_argument('--workers',required=True,type=Path)
    p.add_argument('--suite',choices=('normal','easy','power-loss'),default='normal')
    args=p.parse_args();run(args.root,args.workers,args.suite)
