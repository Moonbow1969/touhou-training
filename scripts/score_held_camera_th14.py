"""Score matched original game outcomes and script ticks on the unseen TH14 trial."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.checkpoint_storage import digest
from scripts.score_native_full_actions import wilson


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--arms',nargs=2,required=True);p.add_argument('--first-seed',type=int,required=True)
    p.add_argument('--n',type=int,default=6);a=p.parse_args();rows={};a.out.mkdir(parents=True,exist_ok=True)
    for arm in a.arms:
        group=[]
        for seed in range(a.first_seed,a.first_seed+a.n):
            folder=a.root/f'v26th14-{arm}-{seed}'
            j=json.loads((folder/'report.json').read_bytes());trace=folder/'native.npz'
            if j['seed']!=seed or j['arm']!=arm or j['continued'] or j['terminal']!='game_over' or j['terminal_spares']!=-1:
                raise ValueError('Missing genuine game-over stock')
            if digest(trace)!=j['trace_sha256'] or digest(folder/'terminal.png')!=j['terminal_image_sha256']:
                raise ValueError('Changed original credit')
            with np.load(trace,allow_pickle=False) as z:
                y=z['action'];states=z['native'];lp=z['logp']
                if len(y)!=j['steps'] or not ((states[0,2:6]==[1,2,3,2]).all()):raise ValueError('Invalid opening')
                peak=int(states[states[:,2]==1,1].max())
                # Record compact actions and verified original resources for every decision.
                small=a.out/(folder.name+'.npz');np.savez_compressed(small,action=y,logp=lp,native=states)
                changes=float(np.mean(y[1:,0]!=y[:-1,0]))
                # A stock transition is linked to X only within two observed four-tick controls.
                previous_x=(y[:-1,3]!=0)
                recent=previous_x|np.r_[False,previous_x[:-1]] if len(previous_x) else previous_x
                spends=int(((np.diff(states[:,4])==-1)&(np.diff(states[:,3])==0)&recent).sum())
            photo=folder/'settled.png'
            if not photo.is_file():photo=folder/'terminal.png'
            image=a.out/(folder.name+'.jpg')
            with Image.open(photo) as im:im.crop((0,0,650,514)).convert('RGB').save(image,quality=85)
            retry_log=folder/'technical-attempts.jsonl'
            retries=[json.loads(v) for v in retry_log.read_text().splitlines() if v.strip()] if retry_log.exists() else []
            group.append({**j,'technical_retries':len(retries),'retry_reasons':[v.get('error',v.get('reason')) for v in retries],
                          'max_observed_stage1_tick':peak,'actual_move_change_rate':changes,
                          'same_life_single_bomb_spends_near_x':spends,'bomb_command_windows':int(y[:,3].sum()),
                          'native_end_image':image.name,'end_capture_phase':photo.stem,
                          'state_columns':['absolute_tick','stage_tick','stage','spares','bombs','game_mode','x','y'],
                          'compact_trace_sha256':digest(small)})
        rows[arm]=group
    left,right=(rows[k] for k in a.arms)
    if any(x['host']!=y['host'] or x['seed']!=y['seed'] for x,y in zip(left,right)):raise ValueError('Unmatched pair')
    dif=np.array([y['max_observed_stage1_tick']-x['max_observed_stage1_tick'] for x,y in zip(left,right)])
    result={'arms':{},'paired_stage1_tick_differences':dif.tolist(),'mean_paired_stage1_tick_difference':float(dif.mean()),
            'paired_bootstrap95_stage1_ticks':np.percentile(np.random.default_rng(261126).choice(dif,(20000,len(dif))).mean(1),[2.5,97.5]).tolist()}
    for label,group in rows.items():
        successes=[x for x in group if x['stage2'] is not None]
        result['arms'][label]={'n':len(group),'stage2':len(successes),'stage2_wilson95':wilson(len(successes),len(group)),
              'technical_retries_before_verified_outcomes':sum(x['technical_retries'] for x in group),
              'seeds_with_retries':[x['seed'] for x in group if x['technical_retries']],
              'mean_stage2_bombs':float(np.mean([x['stage2']['bombs'] for x in successes])) if successes else None,
              'mean_stage2_spares':float(np.mean([x['stage2']['spares'] for x in successes])) if successes else None,
              'mean_terminal_spares':0,'mean_terminal_bombs':float(np.mean([x['terminal_bombs'] for x in group])),
              'total_same_life_single_bomb_spends_near_x':sum(x['same_life_single_bomb_spends_near_x'] for x in group),
              'total_bomb_command_windows':sum(x['bomb_command_windows'] for x in group),
              'mean_stage1_tick':float(np.mean([x['max_observed_stage1_tick'] for x in group])),
              'mean_move_change_rate':float(np.mean([x['actual_move_change_rate'] for x in group]))}
    for name,data in [('credits',rows),('paired',result)]:
        (a.out/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
