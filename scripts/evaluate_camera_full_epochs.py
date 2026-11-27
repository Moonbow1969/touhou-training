"""Register the two fixed training-budget endpoints for four-title full-credit evaluation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True,type=Path)
    d=parser.parse_args().data.resolve()
    train=d/'four-title-complete-epochs';root=d/'full-epochs-original';root.mkdir(exist_ok=True)
    seeds={str(t):list(range(1820051+t*100,1820067+t*100)) for t in (10,12,13,17)}
    parent=train/'protocol.json'
    plan=dict(reference_arm='epoch-01',candidate='epoch-20',eval_seeds=seeds,
        rank_by_seed={t:{str(s):2 if s%2 else 3 for s in ss} for t,ss in seeds.items()},
        training_protocol_sha256=digest(parent),
        design='Compare the first and twentieth complete data pass from one 74-stage SFT optimization. Fix both endpoints irrespective of validation accuracy. Both control all four actions directly from the camera. Eight matched pairs per game and difficulty, 128 independent stock starts in total. Same host and build per pair and balanced order. Native RNG is uncontrolled. Keep physical terminals and separate TH12 complete-control pairs. No continues, aim correction, emergency Bomb or geometric steering.',
        primary='Stage 2 arrival and, when reached, remaining resources; additionally report terminal lives, Bombs or Power and per-game original progress. Retain all technical attempts and native provenance.')
    freeze(root/'camera-protocol.json',plan)
    models={}
    for arm in (plan['reference_arm'],plan['candidate']):
        path=train/arm/'report.json'
        while not path.is_file() or not (train/arm/'policy.ts').is_file():time.sleep(10)
        v=json.loads(path.read_bytes())
        if v['torchscript_sha256']!=digest(train/arm/'policy.ts') or v['protocol']!=digest(parent) or v['epoch']!=int(arm[-2:]):
            raise ValueError('Changed original epoch checkpoint')
        models[arm]=v['torchscript_sha256']
    freeze(root/'checkpoint-binding.json',dict(protocol=digest(root/'camera-protocol.json'),models=models))
    subprocess.run([sys.executable,'-u','-m','scripts.score_multigame_followup','--data',str(d),'--study',root.name],check=True)


if __name__=='__main__':main()
