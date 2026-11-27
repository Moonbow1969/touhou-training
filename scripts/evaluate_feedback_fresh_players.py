"""Score frozen direction-feedback policies on independently registered TH17 retail players."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.audit_feedback_image_alignment import metrics
from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze
from scripts.train_human_camera_policy import load_replay


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    d=p.parse_args().data.resolve()
    source=d/'four-title-direction-feedback'
    root=source/'fresh-th17-players';root.mkdir(exist_ok=True)
    held=d/'human-retail17-wave2-held'
    roster=held/'frozen-source-roster.json'
    arm_files={a:source/a/'policy.ts' for a in ('image','feedback')}
    comparison=source/'image-alignment/report.json'
    protocol=dict(roster_sha256=digest(roster),models={a:digest(f) for a,f in arm_files.items()},
                  action_only_control_sha256=digest(comparison),
                  primary='Participant-macro difference in fixed-image autonomous NLL on actual direction changes, feedback minus image. Compute each participant using all their qualified stage decisions. Report every participant, stage and rank. Number of independent people is at most four.',
                  secondary='Overall and change, start and moving-turn accuracy and conditional NLL. Report the frozen action-only control and qualified denominators. Previous commands come from the human for conditional scoring, and from the exact predicted action marginal for autonomous scoring. All models use identical accepted human states.',
                  exclusions='Independently adjudicate all eight source stages on original replay synchronization. Attribute excluded stages to the source reason. If any preassigned author has no qualified stage, report the surviving pilot result and the planned four-author endpoint as incomplete. Cohort is also used by a separately registered human SFT data expansion.')
    if protocol['roster_sha256']!='ca1e357eb8e0809354ce5c1d5f48101e8995f6feb1f089a0aee8ac9eacabe1bd':
        raise ValueError('Changed fresh human roster')
    freeze(root/'protocol.json',protocol)
    enrolled={(r['sha256'],s):r for r in json.loads(roster.read_bytes())['sources'] for s in r['acquisition_stages']}
    if len(enrolled)!=8 or len({r['player_group'] for r in enrolled.values()})!=4:
        raise ValueError('Unexpected independent cohort')
    seal=held/'held-original-synchronized-frozen.json'
    while not seal.exists():
        time.sleep(30)
    cohort=json.loads(seal.read_bytes())
    all_rows=cohort['qualified']+cohort['excluded']
    if len(all_rows)!=8 or {(r['replay_sha256'],r['stage']) for r in all_rows}!=set(enrolled):
        raise ValueError('Incomplete original synchronization adjudication')
    if any(not r['reason'] for r in cohort['excluded']):
        raise ValueError('Missing technical exclusion reason')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    nets={a:torch.jit.load(str(f),map_location=device).eval() for a,f in arm_files.items()}
    table=np.asarray(json.loads(comparison.read_bytes())['transition_probability'])
    groups={}
    stages={}
    with torch.inference_mode():
        for q in cohort['qualified']:
            selected=enrolled[q['replay_sha256'],q['stage']]
            path=(d/q['path']).resolve();path.relative_to(d)
            witness=(d/q['original_sync_evidence_path']).resolve();witness.relative_to(d)
            if (q['split']!='held' or digest(witness)!=q['original_sync_evidence_sha256'] or
                    digest(path)!=q['archive_sha256'] or selected['player_group']!=q['player_group']):
                raise ValueError('Changed eligible held original')
            row=load_replay(path)
            if row['title']!='TH17' or row['difficulty']!=selected['difficulty']:
                raise ValueError('Mismatched title or native difficulty')
            clock=row['native_frame'];act=row['action'][:,0];prev=row['prior'][:,0]
            keep=np.flatnonzero((np.diff(clock)==4)&(prev[1:]==act[:-1]))+1
            if not len(keep):
                raise ValueError('No consecutive held inputs')
            stage={'action_only':metrics(np.broadcast_to(table,(len(keep),9,9)),act[keep],prev[keep],clock[keep])}
            for arm,net in nets.items():
                matrices=[]
                for ids in np.array_split(keep,(len(keep)+63)//64):
                    x=torch.from_numpy(row['raw'][ids]).to(device).float()/255
                    z=net.features(x);movement=net.head(z)[:,:9]
                    logits=[net.residual(z,movement,torch.full((len(ids),),v,device=device,dtype=torch.long)).softmax(1)
                            for v in (range(9) if arm=='feedback' else (4,))]
                    matrices.append(torch.stack(logits,1).expand(-1,9,-1).cpu().numpy())
                stage[arm]=metrics(np.concatenate(matrices),act[keep],prev[keep],clock[keep])
            pilot=selected['player_group']
            groups.setdefault(pilot,[]).append(stage)
            stages[q['archive_sha256']]=dict(pilot=pilot,rank=selected['difficulty'],stage=q['stage'],metrics=stage)
    per_pilot={}
    for pilot,parts in groups.items():
        scores={}
        for arm in ('feedback','image','action_only'):
            cells=[s[arm]['autonomous']['change'] for s in parts]
            n=sum(c['rows'] for c in cells)
            if n:
                scores[arm]=sum(c['nll_sum'] for c in cells)/n
        if len(scores)==3:
            per_pilot[pilot]=dict(nll=scores,feedback_minus_image=scores['feedback']-scores['image'])
    result=dict(protocol_sha256=digest(root/'protocol.json'),acquisition_sha256=digest(seal),eligible=len(stages),
                planned_stages=8,planned_people=4,eligible_people=len(per_pilot),complete_primary=len(per_pilot)==4,
                macro_autonomous_change_nll_difference=float(np.mean([x['feedback_minus_image'] for x in per_pilot.values()])) if per_pilot else None,
                excluded=cohort['excluded'],per_author=per_pilot,per_stage=stages)
    (root/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
