"""Measure the temporal resolution of human controls in the actual four-title visual SFT set."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from scripts.checkpoint_storage import digest
from scripts.train_causal_replay_history import paths
from scripts.train_five_original_games import freeze
from scripts.train_human_camera_policy import camera_labels, movement_direction


def score_chunks(words,focus_bit,size):
    controls=np.stack((movement_direction(words.ravel()).reshape(words.shape), (words&focus_bit)!=0, (words&1)!=0,(words&2)!=0),axis=-1)
    target=controls.reshape(-1,size,4)
    action=camera_labels(words.reshape(-1,size),focus_bit)
    misses=(target!=action[:,None,:]).sum(axis=(0,1))
    # Best constant direction inside each subwindow is an oracle that sees its future inputs.
    best=np.maximum.reduce([(target[:,:,0]==k).sum(axis=1) for k in range(9)])
    changes=controls[:,1:,0]!=controls[:,:-1,0]
    blocked=[step for step in range(1,4) if step%size]
    return dict(native_ticks=int(words.size),windows=len(words),duration=size,
        direction_mismatched_ticks=int(misses[0]),focus_mismatched_ticks=int(misses[1]),
        shot_mismatched_ticks=int(misses[2]),bomb_mismatched_ticks=int(misses[3]),
        optimal_constant_direction_mismatched_ticks=int((size-best).sum()),
        four_tick_windows_with_internal_turn=int(changes.any(axis=1).sum()),
        turns_between_ticks_inside_four=int(changes.sum()),
        turns_inside_action_hold=int(changes[:,[s-1 for s in blocked]].sum()) if blocked else 0,
        positive_bomb_ticks=int(controls[:,:,3].sum()),issued_bomb_ticks=int(action[:,3].sum()*size))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    d=p.parse_args().data.resolve()
    study=d/'ranked-multigame-sft/four-title-expanded-twelve'
    root=study/'native-control-resolution';root.mkdir(exist_ok=True)
    rows=json.loads((study/'report.json').read_bytes())['train']
    lookup=paths(d)
    extra=json.loads((d/'human-retail17-wave2/train-original-synchronized-frozen.json').read_bytes())
    lookup.update({r['archive_sha256']:d/r['path'] for r in extra['qualified']})
    plan=dict(sft_training_report=digest(study/'report.json'),durations=[1,2,4],
        population='All usable four-native-tick TRAIN windows in the fitted four-title visual learner. Inspect all four native human control words stored at each observation, preserving exact original per-title focus bits. Compare the standard movement, focus and shot on the first tick and any Bomb in the interval, held for 1, 2 or 4 ticks. Report both the native label temporal mismatch and a future-aware oracle which chooses the optimal constant direction in each interval. Neither quantity is a prediction error, original survival outcome or an estimate of new images available at subwindow times.')
    freeze(root/'protocol.json',plan)
    cells=defaultdict(lambda:defaultdict(int));per_stage=[]
    for row in rows:
        path=lookup[row['sha256']]
        if digest(path)!=row['sha256']:raise ValueError('Human training source changed')
        meta=json.loads(path.with_suffix('.json').read_bytes());title=row['title']
        rank=str(meta.get('difficulty') or meta['mode']).split()[-1].title()
        focus=4 if title=='TH10' else 8
        with np.load(path,allow_pickle=False) as z:
            frame=z['native_frame'].reshape(-1)
            words=z['input_chunk']
            source_action=z['action']
            keep=z['stable'].astype(bool) if 'stable' in z else np.ones(len(frame),bool)
            idx=np.arange(len(frame))
            keep&=(idx>=1)&(frame-np.roll(frame,1)==4)
            if 'past_input_chunk' not in z:
                keep&=(idx>=2)&(frame-np.roll(frame,2)==8)
        if not np.array_equal(camera_labels(words,focus),source_action) or keep.sum()!=row['decisions']:
            raise ValueError('Disagreement with the frozen visual training eligibility and labels')
        stage={}
        for size in (1,2,4):
            counts=score_chunks(words[keep],focus,size)
            for k,v in counts.items():
                if k!='duration':cells[(title,rank,size)][k]+=v
            stage[str(size)]=counts
        per_stage.append(dict(name=row['name'],player=row['group'],title=title,difficulty=rank,sha256=row['sha256'],hold_ticks=stage))
    results={f'{t}-{r}-{s}':dict(c) for (t,r,s),c in sorted(cells.items())}
    if len(results)!=24 or sum(v['windows'] for k,v in results.items() if k.endswith('-4'))!=148415:
        raise ValueError('Missing a registered high-difficulty source')
    freeze(root/'report.json',dict(protocol_sha256=digest(root/'protocol.json'),cells=results,stages=per_stage))
    for k,v in results.items():
        if k.endswith('-4'):
            print(k,v['direction_mismatched_ticks']/v['native_ticks'],v['optimal_constant_direction_mismatched_ticks']/v['native_ticks'],flush=True)


if __name__=='__main__':
    main()
