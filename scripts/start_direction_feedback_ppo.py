"""Continue the four-game human feedback policy with fresh two-title on-policy PPO."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    d=p.parse_args().data.resolve()
    root=d/'direction-feedback-original-ppo'
    root.mkdir(exist_ok=True)
    movement=d/'four-title-direction-feedback'
    actor=movement/'feedback'
    visual=d/'specialist-vs-shared/merged-equal/camera-policy.pt'
    comparison=d/'four-title-direction-feedback-native'
    original=json.loads((comparison/'camera-protocol.json').read_bytes())
    if not json.loads((movement/'report.json').read_bytes())['original_test_gate'] or digest(actor/'policy.ts')!=original['policies']['feedback']:
        raise ValueError('The original H1 weight failed its offline qualification')
    train={str(t):list(range(1530000+t*100+1,1530000+t*100+17)) for t in (10,17)}
    held={str(t):list(range(1530000+t*100+51,1530000+t*100+67)) for t in (10,17)}
    rule=dict(study='Full-action original PPO from the four-title feedback SFT',
              games={'TH10':'original 0.02a','TH17':'original publisher trial'},
              train_seeds=train,eval_seeds=held,rank_by_seed={t:{str(s):2 if s%2 else 3 for s in train[t]+held[t]} for t in train},
              behavior_sha256=digest(actor/'policy.ts'),camera_sft_sha256=digest(visual),
              residual_sha256=digest(movement/'feedback.ts'),previous_direction_input=True,
              reference_arm='feedback',candidate='feedback-ppo',balanced_title_rank=True,learning_rates=[2e-5,1e-5,5e-6,2.5e-6],
              objective='Test one four-epoch full-policy update on the qualified human SFT, independent of whether sampling the SFT by itself improved original-game survival. All four action heads, camera and history residual receive gradients. Restore the human camera and movement head. Initialize critic at zero and verify all original action probabilities on the pre-action images and transmitted direction.',
              collection='Two works, two difficulties, eight complete fresh stock training games per cell. H1 direction is neutral on reset then the previously executed action. Keep native action cadence and rewards. Exclude all prior 151 held starts and all 153 held starts from gradients.',
              evaluation='Eight new matched seeds per work and difficulty. Same stock start, host, build and physical lane per pair, order balanced within difficulty. Report Stage 2, last Stage 1 native tick, inventory and casts. Original RNG can differ within pairs. Resources use TH10 Power and TH17 last playable Bombs.')
    protocol=root/'camera-protocol.json'
    freeze(protocol,rule)
    print('Registered 32 original training games and 32 held pairs',digest(protocol),flush=True)
    subprocess.run([sys.executable,'-u','-m','scripts.run_registered_camera_update',
                    '--root',root,'--parent',actor,'--from-sft','--feedback-camera',visual,'--direction-residual',movement/'feedback.ts',
                    '--protocol',protocol],check=True)


if __name__=='__main__':
    main()
