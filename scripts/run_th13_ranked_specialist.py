"""Compare broad and high-rank human SFT at native Hard/Lunatic in original TH13."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import zipfile
import numpy as np

from scripts.collect_th10_full_action import save
from scripts.read_th13_trial_ranking import scan
from scripts.run_th10_camera_cohort import digest
from scripts.run_th13_accelerator_camera import run as collect
from scripts.score_native_stage_clock import summarize as clock_score
from scripts.score_original_camera_run import verify


def score(study,plan):
    native=study/'native13';out=study/'evaluation'
    for k in ('action-traces','endpoints'):(out/k).mkdir(parents=True,exist_ok=True)
    ranks=plan['rank_by_seed']['13'];names=[plan['reference_arm'],plan['candidate']]
    pictures=scan(native);by_folder={Path(r['path']).parent.name:r for r in pictures['rows']}
    rows=[];archive=study/'th13-ranked-native-logs.zip';temp=archive.with_suffix('.staging')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for seed in plan['eval_seeds']['13']:
            rank=ranks[str(seed)]
            for name in names:
                tag=f'held-{name}-{seed}';folder=native/tag;p=folder/'receipt.json'
                log=json.loads(p.read_bytes());reading=by_folder[tag]
                assert log['difficulty']==rank and log['seed']==seed and log['host']=='accelerator'
                assert log['policy_sha256']==plan['checkpoints'][name] and log['terminal'] and not log['continues']
                data=np.fromfile(folder/'steps.bin','<f4').reshape(-1,143)
                assert len(data)==log['steps']==log['difficulty_verified_native_samples']
                assert list(data[0,[125,127,129,141]])==[1,2,2,rank] and (data[:,141]==rank).all()
                assert digest(folder/'steps.bin')==log['native_difficulty_trace_sha256']
                row=verify(13,p,log,folder/'rollout.npz',name,out)
                first=np.flatnonzero(data[:,136]>0)
                row.update(difficulty=rank,native_rank_samples=len(data),rank_evidence_sha256=log['native_difficulty_trace_sha256'],
                    original_menu_sha256=digest(folder/'selected-difficulty.png'),first_native_loss_tick=int(data[first[0],135]) if len(first) else None,
                    original_score=int(data[-1,131])*10,ranking_visual_stage=reading['readout'],ranking_matches_native=reading['matches_native'],
                    native_key_matches=log['native_key_match'],native_start=log['start'],native_finish=log['finish'])
                rows.append(row)
                for n in ('steps.bin','events.tsv','receipt.json','ranking.png','selected-difficulty.png','actor.log','final-stock.bin'):
                    z.write(folder/n,f'{tag}/{n}')
        analyses={}
        for rank in (2,3):
            sample=[r for r in rows if r['difficulty']==rank]
            groups={}
            for name in names:
                arm=[r for r in sample if r['arm']==name]
                assert len(arm)==12
                groups[name]=dict(credits=len(arm),stage2=sum(r['stage2_reached'] for r in arm),stage2_resources=[
                    dict(seed=r['seed'],spares=r['stage2_spares'],bombs=r['stage2_bombs']) for r in arm if r['stage2_reached']],
                    corroborated_rankings=sum(r['ranking_matches_native'] is True for r in arm),
                    mean_final_bombs=np.mean([r['terminal_bombs'] for r in arm]),remaining_lives=[r['terminal_spares'] for r in arm],
                    casts=sum(r['effective_bombs'] for r in arm),mean_original_score=np.mean([r['original_score'] for r in arm]),
                    native_key_matches=sum(r['native_key_matches'] for r in arm),decisions=sum(r['decisions'] for r in arm))
            analyses[rank]=dict(primary=clock_score(sample,13,names),groups=groups)
        save(out/'credits.json',rows);save(out/'ranking-readout.json',pictures);save(out/'paired-by-rank.json',analyses)
        for p in (study/'protocol.json',native/'rank-launcher.json',out/'credits.json',out/'ranking-readout.json',out/'paired-by-rank.json'):
            z.write(p,p.relative_to(study).as_posix())
    with zipfile.ZipFile(temp) as z:assert z.testzip() is None
    temp.replace(archive)


def run(root,workers,initial):
    root=root.resolve();study=root/'ranked-multigame-sft/held-th13';native=study/'native13'
    plan=json.loads((study/'protocol.json').read_bytes());models={a:root/p for a,p in plan['arms'].items()}
    seeds=plan['eval_seeds']['13'];ref,alt=plan['reference_arm'],plan['candidate']
    assert len(set(seeds))==24 and all(plan['rank_by_seed']['13'][str(s)]==3-(s%2) for s in seeds)
    for arm,path in models.items():assert digest(path)==plan['checkpoints'][arm]
    if initial:
        pilot=[(ref,1191351),(alt,1191364),(alt,1191366)]
        while not all((native/f'held-{a}-{s}/receipt.json').is_file() for a,s in pilot):time.sleep(3)
    for block in range(4):
        for rank in ((2,3) if block%2==0 else (3,2)):
            rows=[s for s in seeds if plan['rank_by_seed']['13'][str(s)]==rank]
            for arm,group in ((ref,rows[:6]),(alt,rows[6:]),(alt,rows[:6]),(ref,rows[6:]))[block:block+1]:
                missing=[s for s in group if not (native/f'held-{arm}-{s}/receipt.json').is_file()]
                if not missing:continue
                collect(SimpleNamespace(qualification=root/'engineering/th13-accelerator',connection_config=workers.resolve(),
                    checkpoint=models[arm],output=native,role='held',arm=arm,seeds=missing,workers=3,difficulty=rank))
    score(study,plan)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=Path,required=True)
    p.add_argument('--wait-initial',action='store_true');a=p.parse_args();run(a.root,a.workers,a.wait_initial)
