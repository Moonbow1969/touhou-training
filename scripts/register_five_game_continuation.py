"""Register a fresh original five-title on-policy update."""
import argparse
import json
from pathlib import Path

from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--generation',type=int,choices=(2,3),default=2)
    args=parser.parse_args()
    data=args.data.resolve()
    third=args.generation==3
    source=data/('five-game-generation2' if third else 'five-game-online')
    first=json.loads((source/'camera-protocol.json').read_bytes())
    seed_base=1600000 if third else 1460000
    train={t:[seed_base+100*int(t)+i for i in range(1,33 if third else 17)] for t in first['train_seeds']}
    held={t:[seed_base+100*int(t)+i for i in range(51,67)] for t in train}
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
    if third:
        plan.update(name='Third five-title PPO with 16 original credits per title and rank',
                    reference_arm='five-ppo2',candidate='five-ppo3',
                    question='How reproducible are the policy gradients across disjoint original credits at the '
                             'same frozen five-title checkpoint, and does the resulting PPO advance ordinary stock games?',
                    training='160 new complete physical training credits, 16 per title and difficulty. Restore the '
                             'second-generation actor and critic. Verify original behavior and native terminal evidence. '
                             'All ten cells receive equal optimizer weight. Retain the original reward and four PPO '
                             'epochs. More decisions produce more optimizer minibatches. This update measures the '
                             'combined continuation, not a causal effect of batch size on game performance.',
                    gradient_diagnostic='Before fitting, compare same-cell initial policy gradients across disjoint '
                                        'credit groups (2, 4 or 8 credits per group). Estimate consistency at a fixed '
                                        'behavior model using the same 16 registered credits per cell. Raw gameplay, '
                                        'not repeated optimizer updates, defines the independent unit.')
    root=data/f'five-game-generation{args.generation}';root.mkdir(exist_ok=True)
    freeze(root/'camera-protocol.json',plan)
    if third:
        original=root/'camera-protocol.json'
        terminal=source/'th12-terminal-consumption-registration.json'
        previous=json.loads(terminal.read_bytes())
        freeze(root/terminal.name,dict(original_protocol_sha256=digest(original),
                                      prior_technical_standard_sha256=digest(terminal),
                                      scope='TH12 TRAIN and HELD original fatal-tick control verification',
                                      inherited_standard=previous))
        rule_file=source/'th12-train-technical-amendment.json'
        earlier=json.loads(rule_file.read_bytes())
        rule={k:earlier[k] for k in ('name','applicable_title','role','reserve_stride','max_reserve_attempts','eligibility','output')}
        rule.update(original_protocol_sha256=digest(original),
                    terminal_consumption_registration_sha256=digest(root/terminal.name),
                    original_seeds=train['12'], prior_rule_sha256=digest(rule_file),
                    registration='Before the first physical training attempt in this new cohort.')
        freeze(root/rule_file.name,rule)
    print(root,digest(root/'camera-protocol.json'),flush=True)


if __name__=='__main__':
    main()
