"""Verify two pure camera-policy pilot credits on native TH10 Hard and Lunatic."""
import argparse
import csv
import gzip
import json
from pathlib import Path

from scripts.collect_th10_full_action import save
from scripts.package_th10_credit_logs import package
from scripts.run_th10_camera_cohort import checkpoint,digest,verify
from scripts.score_original_camera_run import verified_th10_entry


def run(root):
    model_path=root/'scaled-human-sft/cnn-new-24/policy/policy.ts'
    model=checkpoint(model_path);records={}
    sources=[]
    for rank,name,seed in [(2,'hard',90001002),(3,'lunatic',90001003)]:
        folder=root/'engineering'/f'th10-{name}';credit=folder/f'momentum-{seed}'
        record=json.loads((credit/'report.json').read_bytes())
        assert digest(model_path)==record['policy_sha256']==json.loads((model_path.parent/'report.json').read_bytes())['torchscript_sha256']
        assert record['terminal'] and record['continues']==0
        assert record['difficulty']==rank and record['opening_native_character']==0 and record['opening_native_shot']==0
        with (credit/'opening.csv').open() as f:opening=next(csv.DictReader(f))
        assert [int(opening[k]) for k in ('stage','difficulty','character','shot','lives','score')]==[1,rank,0,0,2,0]
        with gzip.open(credit/'native.csv.gz','rt') as f:states=list(csv.DictReader(f))
        assert len(states)==record['native_samples']>0 and {int(s['difficulty']) for s in states}=={rank}
        stage2,native_sha,resources=verified_th10_entry(record,credit);assert stage2 is not None and native_sha
        _,errors=verify(folder,model_path,model,[seed],'engineering',write=False,host='momentum',difficulty=rank)
        records[name]=dict(seed=seed,host='momentum',native_difficulty=rank,native_samples=len(states),
            opening=dict(stage=1,character=0,shot=0,spares=2,raw_score=0),model_sha256=record['policy_sha256'],
            source_bundle=record['source']['bundle'],native_header_sha256=record['stage2_evidence']['state_header_sha256'],
            stock_source_sha256=record['stage2_evidence']['stock_source_sha256'],native_csv_sha256=native_sha,
            original_menu_sha256=digest(credit/f'{name}-rank.png'),game_sha256=record['game_sha256'],
            steps=record['steps'],native_key_matches=record['native_inputs_matching'],terminal=record['native_outcome'],
            reached_stage2=stage2,entry_spares=resources[0],entry_power=resources[2],final_spares=record['final_lives'],
            final_power=record['final_power'],confirmed_casts=len(record['successful_bombs']),score=record['raw_score'],
            original_camera_sha256=digest(credit/'original-pixels.npz'),camera_replay=errors[0])
        sources.append(folder)
    out=root/'engineering/th10-high-ranks';out.mkdir(exist_ok=True)
    package(out/'original-pilot-evidence.zip',sources)
    save(out/'qualification.json',dict(qualified=True,game='TH10 official 0.02a original story trial',
        method='Unmodified 38-pilot sampled full-action camera policy. Select Hard with one Down or Lunatic with two Down presses from verified original Rank Select. Require original Reimu A stock start, continuous rank telemetry, qualified Stage 1/2 channel, native terminal and camera-logp parity.',
        credits=records,evidence_sha256=digest(out/'original-pilot-evidence.zip')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);run(parser.parse_args().root)
