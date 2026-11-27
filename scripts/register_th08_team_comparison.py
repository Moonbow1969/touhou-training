"""Register the ordinary-stock four-team comparison before loading four-team evaluation outcomes."""
import argparse
from pathlib import Path

from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    data=p.parse_args().data.resolve()
    root=data/'th08-multiteam-online'
    reference='six-ppo'
    candidate='six-four-teams'
    roster=[]
    for team in ('Border','Magic','Scarlet','Ghost'):
        for rank in ('Hard','Lunatic'):
            for i in range(8):
                roster.append(dict(seed=1720801+len(roster),team=team,rank=rank,
                    arms=[reference,candidate] if i%2==0 else [candidate,reference]))
    freeze(root/'team-held-registration.json',dict(
        title=8,study='Six-title original PPO with one or four TH08 training teams',
        reference_arm=reference,candidate=candidate,
        reference_sha256=digest(data/'six-game-online/joint-six/policy/policy.ts'),
        candidate_path='joint-six-four-teams/policy/policy.ts',
        fit_registration=digest(root/'central-fit-protocol.json'),
        host='Entropy',roster=roster,
        assignment='Reserve the same physical host and lane, original game build, team, and sampled policy seed within each pair. Native game RNG is uncontrolled. Begin after the previously registered 171 training tickets. Counterbalance arm order within each team and difficulty.',
        primary='Reach Stage 2 on a full ordinary credit with zero continues. Preserve an indeterminate original rank or stage, with identified success bounds for each eight-credit arm. Report all assigned including technical failures, per team and rank.',
        secondary='Terminal spare lives and last playable Bombs, movement and button counts, observed gameplay duration and policy decisions. Report paired intervals and the number of credit pairs with complete evidence.',
        retry='Pre-input entry failures retain their attempt logs and may retry the same seed. Preserve every launched post-input credit, including indeterminate outcomes. No outcome-based score replacements.'))


if __name__=='__main__':
    main()
