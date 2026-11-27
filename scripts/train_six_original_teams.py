"""Fit comparable six-title PPO policies using one or four selectable original TH08 teams."""
import argparse
from collections import Counter
from pathlib import Path
from scripts.train_five_original_games import freeze
from scripts.train_six_original_games import digest, optimize, read, wait


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    data=parser.parse_args().data.resolve()
    root=data/'th08-multiteam-online'
    team_plan=read(root/'protocol.json')
    border=data/'six-game-online'
    border_plan=read(border/'protocol.json')
    parent=data/'five-game-generation2/policy'
    five_manifest=data/'five-game-generation3/corrected/credits.json'
    registration=root/'central-fit-protocol.json'
    plan=dict(name='Six-title PPO with four original TH08 teams',parent_actor=digest(parent/'policy.ts'),
              parent_actor_critic=digest(parent/'camera-ppo.pt'),five_title_manifest=digest(five_manifest),
              multiteam_registration=digest(root/'protocol.json'),border_registration=digest(border/'protocol.json'),
              common_border_slots=border_plan['first_th08_fit']['training'],registered_training_slots=team_plan['train'],
              design='Reuse exactly the 160 original five-title training credits, the first eight Border Team credits, initial parameters, optimizer and 12 equally weighted title/rank cells. In the comparison policy replace 24 other Border Team games with eight each for Magic, Scarlet and Ghost. Each TH08 team contributes four Hard and four Lunatic original games across the complete four-team fit. Outcome-dependent selection is forbidden. The host and physical timing can differ between team collectors. The number of decisions may differ even at equal numbers of original games.',
              evaluation='Keep training data separate from ordinary-start held games. Compare policies within original host, team, difficulty and sampled policy seed. Freeze the held roster before measuring the four-team policy.')
    if plan['parent_actor']!=team_plan['policy_sha256'] or plan['parent_actor']!=border_plan['parent_actor']:
        raise ValueError('The team collector must use the same frozen five-game behavior')
    freeze(registration,plan)
    q=root/'qualified-train.json'
    wait(q)
    alternate=read(q)['rows']
    if len(alternate)!=24:
        raise ValueError('Expected all 24 qualified allocations')
    first=read(border/'first-eight/optimizer-input/credits.json')
    if len(first)!=8 or Counter(r['difficulty'] for r in first)!={2:4,3:4}:
        raise ValueError('Unbalanced first-eight Border Team credits')
    rows=read(five_manifest)
    if len(rows)!=160 or len({(r['title'],r['seed']) for r in rows})!=160:
        raise ValueError('Expected all five-game credits')
    sources=[(r,five_manifest.parent/r['file']) for r in rows]+[(r,border/'first-eight/optimizer-input'/r['file']) for r in first]
    teams=Counter()
    available={r['seed']:r for r in team_plan['train']}
    seen=set()
    for row in alternate:
        slot=row.get('slot',row['seed'])
        scheduled=available[slot]
        if slot in seen or row['seed'] not in (slot,slot+10000,slot+20000) or row['title']!=8:
            raise ValueError('Duplicate or ineligible TH08 physical credit')
        seen.add(slot)
        rank={'Hard':2,'Lunatic':3}[scheduled['rank']]
        if row['difficulty']!=rank or row['team']!=scheduled['team']:
            raise ValueError('Team or rank differs from original registration')
        path=(root/row['file']).resolve()
        path.relative_to(root)
        teams[(row['team'],rank)]+=1
        sources.append((dict(row,file=path.name),path))
    if set(teams)!={(team,rank) for team in ('Magic','Scarlet','Ghost') for rank in (2,3)} or set(teams.values())!={4}:
        raise ValueError('Expected four original credits per alternate team and rank')
    optimize(root,sources,parent,plan,'joint-six-four-teams',registration=registration)


if __name__=='__main__':
    main()
