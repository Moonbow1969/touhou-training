"""Report human camera windows and fitted original PPO inputs by game and difficulty."""
import argparse
from collections import Counter
import json
from pathlib import Path
from scripts.checkpoint_storage import digest


def read(p):return json.loads(p.read_bytes())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','public'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();d=a.data.resolve();out=a.public.resolve()
    human=d/'six-title-human-camera-th16';report=read(human/'policy/report.json')
    ledger=d/'learning-curves/hard-lunatic-training-coverage.json';ppo=read(ledger)
    base=read(d/'human-train-six-complete-credits/later-train-complete-roster-frozen.json')
    roster=base['prior47_train']+base['qualified']+read(d/'first-five-title-retail-sft/added-reviewed-th11.json')['players']+read(human/'added-th16.json')['players']
    if [v['archive_sha256'] for v in roster]!=[v['sha256'] for v in report['train']]:raise ValueError('Fitted SFT rows changed')
    counts=Counter()
    for src,row in zip(roster,report['train']):
        meta=read((d/src['path']).with_suffix('.json'))
        rank=(meta.get('difficulty') or meta.get('mode') or meta['declared_rank']).split()[-1].title()
        if rank not in ('Hard','Lunatic'):raise ValueError('Unexpected training difficulty')
        title=row['title']
        counts[title,rank,'windows']+=row['decisions'];counts[title,rank,'stages']+=1
    games=sorted({g for g,r,m in counts}|set(ppo['by_game']))
    lines=['# Which Touhou games have trained a policy?', '',
           'The human columns count distinct synchronized four-tick screen/action windows in the six-title visual SFT. The PPO columns count unique original Hard and Lunatic training games included in completed updates across the main experimental series. A single six-title PPO update also used all six online titles together. Reusing a game for another optimizer or epoch adds zero to these PPO counts.', '',
           '| Game | Human Hard windows | Human Lunatic windows | PPO Hard games | PPO Hard decisions | PPO Lunatic games | PPO Lunatic decisions |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    totals=[0]*6
    for game in games:
        values=[counts[game,rank,'windows'] for rank in ('Hard','Lunatic')]
        for rank in ('Hard','Lunatic'):
            value=ppo['by_game'].get(game,{}).get(rank,{});values.extend([value.get('credits',0),value.get('decisions',0)])
        totals=[a+b for a,b in zip(totals,values)]
        lines.append('| '+game+' | '+' | '.join(f'{v:,}' for v in values)+' |')
    if totals[0]+totals[1]!=sum(x['decisions'] for x in report['train']) or totals[2]+totals[4]!=ppo['credits'] or totals[3]+totals[5]!=ppo['decisions']:raise ValueError('Coverage totals do not reconcile')
    lines.append('| **Total** | '+' | '.join(f'**{v:,}**' for v in totals)+' |')
    lines+=['','TH08 also has separate action-history imitation research. TH16 season-release inputs are preserved in its human data. The six-title visual SFT predicts movement, focus, shooting and Bomb. The ledger counts a five-title SFT-to-PPO training game once the optimizer has consumed it.', '',
            '[Six-title SFT source and weights](six-title-human-camera-th16/README.md) · [PPO input ledger](learning-curves/hard-lunatic-training-coverage.json) · [192 independent original full-stock tests](six-game-return-horizon/all-six-original-results.md) · [Five-title human SFT to online PPO](reviewed-five-human-online/README.md)']
    (out/'training-coverage.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('Coverage verified',totals,flush=True)


if __name__=='__main__':main()
