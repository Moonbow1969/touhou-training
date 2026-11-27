"""Evaluate the first real TH08 update on complete paired Hard and Lunatic stocks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from scripts.checkpoint_storage import digest
from scripts.continue_th08_online_extension import atomic_json, launch, selected_credit
from scripts.train_five_original_games import freeze
from scripts.score_th08_original_ranking import inspect_results


def wait(path):
    while not path.is_file():
        time.sleep(10)


def known_stage_one(path):
    refs=Path(__file__).resolve().parents[1]/'tools/rollout-worker-probe/th08-trial/terminal-stage1-boss.npz'
    with np.load(refs,allow_pickle=False) as src:
        reference=src['edges'];mask=src['mask']
    gray=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError('Missing original terminal photograph')
    chip=gray[25:43,33:125].astype(np.float32)
    feature=(chip-cv2.GaussianBlur(chip,(0,0),1.2))[mask]
    reference=reference[mask]
    corr=float(feature@reference/max(np.linalg.norm(feature)*np.linalg.norm(reference),1e-7))
    return dict(terminal_stage=1 if corr>=.80 else None,native_wriggle_name_similarity=corr,
                stage2=False if corr>=.80 else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('data','assets'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--postcredit-assets',type=Path)
    a=p.parse_args();data=a.data.resolve();assets=a.assets.resolve()
    root=data/'six-game-online';sub=root/'first-eight';out=sub/'evaluation'
    out.mkdir(exist_ok=True,parents=True)
    study=json.loads((root/'protocol.json').read_bytes())
    seeds=study['first_th08_fit']['comparisons']
    qualifiers=Path(__file__).resolve().parents[1]/'tools/rollout-worker-probe/th08-trial'
    arms=['five-ppo2','th08-ppo']
    orders={str(seed):arms if (index//2)%2==0 else arms[::-1] for index,seed in enumerate(seeds)}
    plan=dict(parent_registration_sha256=digest(root/'protocol.json'),ranks={str(s):study['rank_by_seed'][str(s)] for s in seeds},
              order_by_seed=orders,lane='Gamer WSL2 private Wine Xvfb; one lane per rank. Wait for the rank training allocation to complete to avoid adding CPU contention.',
              endpoint='First original continue prompt. Wriggle Nightbug visible on the terminal screen certifies Stage 1 and hence no Stage 2. Other endpoints need independent native stage evidence; report unknown count and bounds.',
              native_stage1_reference_sha256=digest(qualifiers/'terminal-stage1-boss.npz'),
              technical='Recover verified first terminal before considering missing footage. If either matched credit lacks a verified first terminal, preserve both and try the same controller-seed pair at +10000, then +20000. No selection on survival or resources.',
              summary='Per difficulty report 8 pairs, macro wall-clock duration from opening to first continue, physical decisions, final Bombs, spares and credited stage evidence, plus paired bootstrap intervals.')
    freeze(out/'held-protocol.json',plan)
    if a.postcredit_assets:
        freeze(out/'ranking-addendum.json',dict(original_protocol=digest(out/'held-protocol.json'),
              original_physical_stream=digest(assets/'stream'),post_game_wrapper=digest(a.postcredit_assets/'stream'),
              calibration_source=digest(qualifiers/'score-row-stage-one.npz'),
              scope='After game control ends and all policy keys are released, use the original continue menu to expose the new score row. The physical gameplay stream, language, game version, policy and data split remain as registered.',
              acceptance='Require the original score-table header and a unique dated row with a margin greater than .30 over the other nine baseline entries. The native Stage 1 glyph must correlate above .88 in at least two independent captured ranking frames. Other stage digits require separate original-image adjudication.'))
    wait(sub/'policy/report.json')
    files={'five-ppo2':data/'five-game-generation2/policy/policy.ts','th08-ppo':sub/'policy/policy.ts'}
    digests={k:digest(v) for k,v in files.items()}
    if digests['five-ppo2']!=study['parent_actor'] or digests['th08-ppo']!=json.loads((sub/'policy/report.json').read_bytes())['policy_sha256']:
        raise ValueError('Held checkpoint changed')
    freeze(out/'checkpoint-binding.json',digests)
    torch.set_num_threads(1)
    def lane(rank):
        for s in study['th08_train_seeds']:
            if study['rank_by_seed'][str(s)]==rank:
                wait(root/'selected-training'/f'{s}.json')
        models={k:torch.jit.load(str(path),map_location='cpu').eval() for k,path in files.items()}
        for s in seeds:
            if plan['ranks'][str(s)]!=rank:continue
            target=out/f'pair-{s}.json'
            if target.is_file():continue
            rejected=[]
            for attempt in range(3):
                seed=s+attempt*10000;rows={};bad=[]
                for arm in orders[str(s)]:
                    dest=root/'native08'/f'held-{arm}-{seed}'
                    if not (dest/'receipt.json').exists():
                        runtime=assets
                        if a.postcredit_assets:
                            runtime=a.postcredit_assets.resolve()
                            dest.mkdir(exist_ok=True,parents=True)
                            if not (dest/'original-stream').exists():os.link(assets/'stream',dest/'original-stream')
                            if digest(dest/'original-stream')!=digest(assets/'stream'):
                                raise ValueError('Changed physical gameplay capture')
                        launch(runtime,files[arm],dest,rank,seed,'held')
                    item,reason=selected_credit(root,dest,models[arm],digests[arm],seed,rank,'held',sub/'packed-held'/arm)
                    if reason:
                        bad.append(dict(arm=arm,**reason))
                    else:
                        terminal=root/item['qualified_source']/'terminal.png'
                        item.update(known_stage_one(terminal))
                        ranking=inspect_results(dest)
                        item['original_ranking']=ranking
                        if ranking['stage']==1:
                            item.update(terminal_stage=1,stage2=False)
                        rows[arm]=item
                if bad:
                    rejected.append(dict(seed=seed,technical=bad,qualified_in_incomplete_pair=rows))
                    continue
                atomic_json(target,dict(slot=s,seed=seed,rank=rank,order=orders[str(s)],rows=rows,rejected=rejected))
                break
            else:raise ValueError(f'Unfilled registered pair {s}')
    with ThreadPoolExecutor(2) as pool:
        jobs=[pool.submit(lane,r) for r in ('Hard','Lunatic')]
        for job in jobs:job.result()
    pairs=[json.loads((out/f'pair-{s}.json').read_bytes()) for s in seeds];summary={}
    for rank in ('Hard','Lunatic'):
        these=[p for p in pairs if p['rank']==rank]
        fields={}
        for metric in ('observed_duration_s','decisions','last_playable_bombs','terminal_spares'):
            x=np.array([[p['rows'][arm]['audit'][metric] for arm in arms] for p in these])
            diff=x[:,1]-x[:,0];samples=np.random.default_rng(1630888).choice(diff,(10000,len(diff)),replace=True).mean(1)
            fields[metric]=dict(means=dict(zip(arms,x.mean(0).tolist())),candidate_minus_parent=float(diff.mean()),
                                paired_seed_bootstrap95=np.quantile(samples,[.025,.975]).tolist())
        stages={}
        for arm in arms:
            status=[p['rows'][arm]['stage2'] for p in these]
            lower=status.count(True);unknown=status.count(None)
            stages[arm]=dict(confirmed_stage2=lower,confirmed_stage1=status.count(False),unknown=unknown,
                             stage2_rate_bounds=[lower/len(status),(lower+unknown)/len(status)])
        summary[rank]=dict(n=len(these),metrics=fields,original_stage_evidence=stages)
    freeze(out/'held-results.json',dict(registered=digest(out/'held-protocol.json'),actor_binding=digests,
                                       paired=summary,pairs=pairs))


if __name__=='__main__':
    main()
