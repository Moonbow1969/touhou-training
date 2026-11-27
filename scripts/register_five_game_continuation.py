"""Register fresh original five-title trajectories for a second on-policy update."""
import argparse
import json
from pathlib import Path

from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    data=parser.parse_args().data.resolve()
    source=data/'five-game-online'
    first=json.loads((source/'camera-protocol.json').read_bytes())
    train={t:[1460000+100*int(t)+i for i in range(1,17)] for t in first['train_seeds']}
    held={t:[1460000+100*int(t)+i for i in range(51,67)] for t in train}
    ranks={t:{str(s):2 if s%2 else 3 for s in train[t]+held[t]} for t in train}
    parent=digest(source/'policy/policy.ts')
    if parent!=json.loads((source/'policy/report.json').read_bytes())['policy_sha256']:
        raise ValueError('Changed first generation actor')
    th11=json.loads((data/'th11-original-ppo/stage2-qualification.json').read_bytes())
    plan={k:first[k] for k in ('games','balanced_title_rank','learning_rates','ppo','controls')}
    plan.update(name='Fresh second-generation five-title PPO',reference_arm='five-ppo',candidate='five-ppo2',
                behavior_sha256=parent,parent_actor_critic_sha256=digest(source/'policy/camera-ppo.pt'),
                native_executable_sha256=th11['native_executable_sha256'],
                train_seeds=train,eval_seeds=held,rank_by_seed=ranks,
                question='Does another on-policy update on fresh states visited by the first five-title actor improve original full-credit progression across titles?',
                training='80 new physical full credits, eight per game and rank. Restore the complete previous actor and critic. Replay and verify every original behavior log probability and value. Retain original rewards, four epochs, equal optimizer weight per title and rank. Require corrected continuous TH12 control and original TH11 native stock-event alignment.',
                evaluation='80 fresh pairs. Same original build and physical host within each pair. Alternate execution order where supported. Controller seeds govern action sampling, with uncontrolled original RNG. Report Stage 2 hits, original game overs, spare lives, remaining Bombs or Power and effective casts. Report last playable original stage tick on titles with qualified clocks. Do not treat observation windows or optimizer epochs as independent credits.')
    root=data/'five-game-generation2';root.mkdir(exist_ok=True)
    freeze(root/'camera-protocol.json',plan)
    print(root,digest(root/'camera-protocol.json'),flush=True)


if __name__=='__main__':
    main()
