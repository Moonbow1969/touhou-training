"""Independently rerun original Normal TH10 pairs after native stage-register qualification."""
import argparse
import csv
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.collect_th10_full_action import save
from scripts.package_th10_credit_logs import package
from scripts.run_th10_camera_cohort import digest
from scripts.run_th10_camera_pair import run as paired


def run(root,workers):
    root=root.resolve();workers=workers.resolve()
    calibration=root/'engineering/stage-positive-control/sensor-repair.json'
    proof=json.loads(calibration.read_bytes())
    assert proof['qualified'] and proof['corrected_register']=='0x472b40'
    for experiment,baseline,first in [('v2-online','v2-supervised/joint-v2-3000',981051),
                                       ('attention-online','spatial-policy-study/spatial-attention',1121051)]:
        new=root/(experiment+'-corrected-stage');out=new/'native10';out.mkdir(parents=True,exist_ok=True)
        old=root/experiment/'native10'
        seeds=list(range(first,first+12));reference=root/baseline/'policy.ts';candidate=root/experiment/'policy/policy.ts'
        save(out/'correction-protocol.json',dict(source_campaign=experiment,original_controller_seeds=seeds,physical_host='momentum',
            fresh_original_credits=True,independent_native_game_rng=True,overwrites_historical_credit=False,
            corrected_stage_register=proof['corrected_register'],additional_negative_control_register=proof['legacy_register'],
            original_title_witness_sha256=digest(calibration),cadence_ms=120,
            actions='All four camera-policy heads, no geometry, aim intervention or automatic Bomb. Original Normal full stock, no Continue.'))
        paired(SimpleNamespace(out=out,workers=workers,host='momentum',difficulty=1,stage_sensor='qualified-v2',
            reference_name='held-sft',candidate_name='held-ppo',reference=reference,candidate=candidate,
            seeds=seeds,concurrency=3,candidate_history=False))
        receipts=[]
        for arm in ('held-sft','held-ppo'):
            for seed in seeds:
                folder=out/arm/f'momentum-{seed}';r=json.loads((folder/'report.json').read_bytes())
                with gzip.open(folder/'native.csv.gz','rt') as f:states=list(csv.DictReader(f))
                assert r['stage_sensor_validated'] and r['stage_register']=='0x472b40'
                assert r['terminal'] and not r['continues'] and {v['difficulty'] for v in states}=={'1'}
                assert int(states[0]['stage'])==1 and r['stage2_reached']==any(int(v['stage'])>=2 for v in states)
                older=old/arm/f'momentum-{seed}'/'report.json'
                if older.is_file():
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
        package(new/'th10-held-native-logs.zip',[out/'held-sft',out/'held-ppo'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path);p.add_argument('--workers',required=True,type=Path)
    args=p.parse_args();run(args.root,args.workers)
