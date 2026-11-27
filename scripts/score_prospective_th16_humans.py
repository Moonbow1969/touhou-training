"""Test zero-shot and trained camera policies on separately registered original TH16 players."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze


def read(path):return json.loads(path.read_bytes())
def wait(path):
    while not path.is_file():time.sleep(30)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data','public'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();d=a.data.resolve();study=d/'six-title-camera-capacity';dest=a.public.resolve()/study.name
    source=d/'th16-visual-expansion/independent-held'
    roster=source/'four-unseen-authors-roster.json';final=source/'held-original-synchronized-frozen.json'
    arms={'five-zero-shot':d/'first-five-title-retail-sft/five-reviewed','six-cnn':study/'cnn','six-attention':study/'attention'}
    design=study/'prospective-th16-comparison.json'
    freeze(design,dict(five_title_actor=digest(arms['five-zero-shot']/'policy.ts'),six_title_training=digest(study/'protocol.json'),
         held_source=roster.relative_to(d).as_posix(),held_outcomes=final.relative_to(d).as_posix(),
         enrollment='Preserve all four registered distinct players across Hard and Lunatic and identify exclusions with source evidence. No held data goes into training. Evaluate only after completion of the entire roster.',
         models=list(arms),outcomes='For each player report overall movements, movement starts and turns, movement log loss, focus, shooting and Bomb. Direction figures are pooled over native four-tick windows. This human screen test supplies no campaign reach rate. The three models produce four common action heads.'))
    dest.mkdir(exist_ok=True);shutil.copy2(design,dest/design.name)
    wait(roster)
    planned=read(roster)['sources']
    roster_binding=dict(original_roster=digest(roster),replays=sorted(v['replay_sha256'] for v in planned))
    if len(set(roster_binding['replays']))!=4:raise ValueError('Four distinct original replay identities required')
    freeze(dest/'prospective-th16-roster.json',roster_binding)
    wait(final);wait(dest/'matched-inputs.json')
    results=read(final);rows=results['qualified'];excluded=results['excluded']
    if (len(rows)+len(excluded)!=4 or results['split']!='held' or any(v['split']!='held' for v in rows)
        or sorted(v['replay_sha256'] for v in rows+excluded)!=roster_binding['replays'] or digest(roster)!=roster_binding['original_roster']):
        raise ValueError('Unexpected prospective allocation')
    paths=[d/v['path'] for v in rows]
    if any(digest(f)!=v['archive_sha256'] for f,v in zip(paths,rows)):raise ValueError('Changed held cameras')
    train_groups={v['group'].casefold() for arm in arms.values() for v in read(arm/'report.json')['train']}
    held_groups={v['player_group'].casefold() for v in rows}
    if len(held_groups)!=len(rows) or held_groups & train_groups:raise ValueError('Player leakage')
    freeze(dest/'prospective-th16-coverage.json',dict(registered=4,original_roster=digest(roster),source_evidence=digest(final),qualified=len(rows),
        excluded=excluded,models={k:digest(v/'policy.ts') for k,v in arms.items()}))
    if paths:
        out=study/'prospective-th16'
        subprocess.run([sys.executable,'-m','scripts.compare_human_camera_policies','--models',*map(str,arms.values()),'--model-names',*arms,
                        '--held',*map(str,paths),'--out',str(out)],check=True)
        shutil.copytree(out,dest/out.name,dirs_exist_ok=True)
    print('Completed original TH16 held data scoring',len(rows),len(excluded),flush=True)


if __name__=='__main__':main()
