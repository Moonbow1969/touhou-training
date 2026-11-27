"""Freeze independent TH10 and TH17 tests of the observed Hard stage-progress signal."""
import argparse
import json
from pathlib import Path
import shutil
from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--public',type=Path,required=True)
    a=p.parse_args();d=a.data.resolve();name='feedback-ppo-independent-replication';s=d/name;s.mkdir(exist_ok=True)
    prior=d/'four-step-feedback-online';plan=json.loads((prior/'camera-protocol.json').read_bytes())
    parent=d/'four-title-four-step-feedback/teacher/policy.ts';child=prior/'policy/policy.ts'
    if digest(parent)!=plan['behavior_sha256'] or digest(child)!=json.loads((prior/'policy/report.json').read_bytes())['policy_sha256']:
        raise ValueError('Unverified prior actors')
    held={};ranks={}
    for t in (10,17):
        start=1870000+100*t
        hard=list(range(start+1,start+25));luna=list(range(start+51,start+59));held[str(t)]=hard+luna
        ranks[str(t)]={**{str(x):2 for x in hard},**{str(x):3 for x in luna}}
    rule=dict(name='Independent TH10 and TH17 full-stock replication of the Hard progress signal',
        reference_arm=plan['reference_arm'],candidate=plan['candidate'],games=plan['games'],controls=dict(geometric_evasion=False, aim_correction=False, auto_bomb=False),eval_seeds=held,rank_by_seed=ranks,
        behavior_sha256=digest(parent),candidate_sha256=digest(child),train_seeds={},previous_direction_input=True,
        original_hypothesis_source=digest(prior/'evaluation/credits.json'),
        checkpoints='Use the frozen teacher-forced movement-feedback SFT and its 64-credit on-policy PPO child. Both policies predict movement, focus, shooting and Bomb. No training or tuning uses replication outcomes.',
        design='New Stage 1 full-stock starts without Continue. Hard has 24 pairs per game. Lunatic has 8 pairs per game. Bind each pair to one host and lane before observing its outcomes. Use balanced randomized first-actor assignment within game, rank and lane. Preserve setup faults, partial attempts and coverage status. Retain both observed endpoints and include all paired seeds after the fixed roster finishes. Native RNG is uncontrolled.',
        primary='Two prespecified Hard outcomes, mean paired difference in last observed native Stage 1 tick, one test per game. If a pair enters Stage 2, retain the highest verified Stage 1 tick at transition and also report the binary Stage 2 result. Compare randomized actor assignments conditional on block restrictions. Use two-sided tests and Holm control across the two Hard comparisons at 0.05. If enumerating all assignments is infeasible, use 200000 valid uniform assignments with RNG seed 1872026 and an add-one correction. Publish effects, paired seed differences and uncertainty even if neither test rejects.',
        secondary='Report native Stage 2 counts and unknown outcomes for every arm and rank, stock at Stage 2, terminal spare lives, last playable Bombs or Power, effective casts, first life-loss tick, progress on Lunatic and paired wins/ties/losses. Treat other comparisons as descriptive. Prior exploratory pairs stay separate.')
    path=s/'camera-protocol.json';freeze(path,rule)
    pub=a.public.resolve()/name;pub.mkdir(exist_ok=True);shutil.copy2(path,pub/path.name)
    print('Registered',sum(len(v) for v in held.values()),'fresh paired seeds',digest(path),flush=True)


if __name__=='__main__':main()

