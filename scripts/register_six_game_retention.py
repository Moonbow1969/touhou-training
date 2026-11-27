"""Register complete original games to measure learning TH08 and retaining two previous titles."""
import argparse
import json
from pathlib import Path
from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True,type=Path)
    d=p.parse_args().data.resolve()
    root=d/'six-game-retention'
    root.mkdir(exist_ok=True)
    old=d/'five-game-generation3'
    six=d/'six-game-online'
    parent=old/'policy/policy.ts'
    old_plan=json.loads((old/'camera-protocol.json').read_bytes())
    games={g:old_plan['games'][g] for g in ('TH10','TH17')}
    games['TH08']='Original standalone trial v0.03, Border Team'
    seeds={str(t):[1690000+100*t+k for k in range(51,67)] for t in (8,10,17)}
    plan=dict(name='Adding TH08 to five-title online PPO',reference_arm='five-ppo3',candidate='six-ppo',
              parent_actor=digest(parent),reference_registration=digest(old/'camera-protocol.json'),
              candidate_registration=digest(six/'protocol.json'),candidate_fit=digest(six/'joint-six/fit-protocol.json'),
              games=games,eval_seeds=seeds,rank_by_seed={t:{str(s):2 if s%2 else 3 for s in ss} for t,ss in seeds.items()},
              training='Both actors start from the same frozen G2 checkpoint and use the same 160 five-game physical trajectories. The six-game fit adds 32 original TH08 credits. It receives more optimizer minibatches after adding this data. The comparison measures the complete training intervention.',
              design='Use TH08 for acquisition, TH10 and TH17 for retention. The selected existing titles previously showed different responses to updates. Eight paired policy seeds per title and difficulty with both independent stock starts on the same original build and physical host. Alternate order within rank. Game RNG is uncontrolled. Each rank lane starts after its pre-existing evaluation commitments.',
              outcome='Within each title/rank primary Stage 2 reach from ordinary full stock. Report unknown or inconclusive stage evidence with bounds. Secondary native stage progress, terminal spare lives, Bombs or Power and verified casts. No game continuation, guide, aimed correction, geometry, forced attack or automatic Bomb. Do not pool original time scales. Retain technical failures and both primary denominators.')
    freeze(root/'camera-protocol.json',plan)

if __name__=='__main__':
    main()
