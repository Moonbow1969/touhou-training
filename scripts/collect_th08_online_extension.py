"""Collect the registered TH08 Hard/Lunatic rows for a fifth-to-sixth-title online extension."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('data','assets'):
        p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args();data=a.data.resolve();assets=a.assets.resolve()
    root=data/'six-game-online';root.mkdir(exist_ok=True)
    parent=data/'five-game-generation2/policy'
    g3=data/'five-game-generation3/camera-protocol.json'
    code=Path(__file__).resolve().parents[1]
    rfile=code/'tools/rollout-worker-probe/th08-trial/rank-glyphs.npz'
    seeds=list(range(1630801,1630833));held=list(range(1630851,1630867))
    plan=dict(parent_actor=digest(parent/'policy.ts'),parent_actor_critic=digest(parent/'camera-ppo.pt'),
              five_title_generation3_protocol=digest(g3),native_edition='TH08 official trial v0.03',
              native_game=digest(assets/'game/th08tr.exe'),native_data=digest(assets/'game/th08tr.dat'),
              key_worker=digest(assets/'stream'),camera_source=digest(code/'scripts/run_th08_trial_camera.py'),
              native_rank_reference=digest(rfile),th08_train_seeds=seeds,
              rank_by_seed={str(s):('Hard' if s%2 else 'Lunatic') for s in seeds+held},
              first_th08_fit=dict(training=seeds[:8],comparisons=held,paired_arms=['five-ppo2','th08-ppo'],
                  evaluation='Hold both complete credits on the same CPU lane and selected rank per controller seed. Alternate arm order inside rank. Original-game RNG is uncontrolled.'),
              joint_six_fit='Use all 32 TH08 camera credits with the original 160 qualified five-title G3 credits. All are on the same immutable five-title actor and critic. Every game/rank receives equal loss weight. Unique physical G3 decisions are reused and reported as such.',
              reward='Observed spare life gain +1, confirmed death -1 including final death. Observed Bomb gains +0.03 and spends -0.03 per unit, exclude native replenishment in a six-decision neighborhood of a life change. No wall-clock survival credit. GAE gamma .999 and lambda .985.',
              stock_qualification='Original rank glyph, two spare stars and three Bomb stars at start, actual continue text with zero spare stars at end, stable inventory for two observations, complete physical keyboard and policy reconstruction required. If any source is technically ineligible, preserve it and report reason; do not select on survival length.',
              first_outcome='Full original credit to first continue prompt. Report terminal spare lives, Bombs, verified life losses, policy decisions and observed time. Stage 2 requires separately checked original stage evidence.',
              engineering_excluded=[1610801,1610802,1620801,1620802])
    freeze(root/'protocol.json',plan)
    def lane(rank):
        for seed in seeds:
            if plan['rank_by_seed'][str(seed)]!=rank:continue
            dest=root/'native08'/f'train-five-ppo2-{seed}'
            if not (dest/'receipt.json').exists():
                subprocess.run([sys.executable,'-u','-m','scripts.run_th08_trial_camera','--assets',str(assets),
                                '--policy',str(parent/'policy.ts'),'--out',str(dest),'--rank',rank,'--seed',str(seed),'--role','train'],check=True)
            receipt=json.loads((dest/'receipt.json').read_bytes())
            if receipt['outcome']!='original_continue_prompt' or receipt['policy_sha256']!=plan['parent_actor']:
                raise ValueError(f'Unqualified original terminal {seed}')
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(lane,r) for r in ('Hard','Lunatic')]
        for job in jobs:job.result()


if __name__=='__main__':
    main()
