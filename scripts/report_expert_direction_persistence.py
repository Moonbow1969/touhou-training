"""Separate held expert direction changes from errors when the expert kept moving the same way."""
import argparse
import json
from pathlib import Path
from scripts.checkpoint_storage import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',required=True,type=Path)
    parser.add_argument('--arm',required=True)
    args=parser.parse_args();source=args.comparison;actor=json.loads(source.read_bytes())[args.arm]
    games={}
    for filename,row in actor['replays'].items():
        game=filename[:4].upper()
        entry=games.setdefault(game,dict(windows=0,stays=0,changes=0,correct_on_changes=0,incorrect_on_stays=0,correct=0))
        metrics=row['aligned'];all_moves=metrics['all'];changes=metrics['change']
        if sum(metrics[event]['n'] for event in ('start','turn','stop'))!=changes['n']:
            raise ValueError('Expert change count disagrees with constituent events')
        stays=all_moves['n']-changes['n']
        missed_stays=stays-(all_moves['correct']-changes['correct'])
        if not 0<=missed_stays<=stays:raise ValueError('Invalid prediction counts')
        for key,value in [('windows',all_moves['n']),('stays',stays),('changes',changes['n']),('correct_on_changes',changes['correct']),('incorrect_on_stays',missed_stays),('correct',all_moves['correct'])]:
            entry[key]+=value
    lines=['# Held movement with the previous human input available', '',
           'On a recorded human demonstration, predicting the previous expert direction is correct on every unchanged direction and wrong on every change. The five-title camera policy receives the playfield image and its recent difference. This comparison separates correctly predicted human changes from choices that interrupt a human hold. The persistence column uses the recorded previous expert action. It measures temporal predictability of the demonstration and does not estimate autonomous survival.', '',
           '| Game | Windows | Correct by repeating previous expert direction | Correct by screen policy | Expert changes correctly predicted | Policy errors on unchanged direction |',
           '|---|---:|---:|---:|---:|---:|']
    for game,v in sorted(games.items()):
        if v['correct']-v['stays']!=v['correct_on_changes']-v['incorrect_on_stays']:raise ValueError('Change/stay decomposition failed')
        lines.append(f"| {game} | {v['windows']:,} | {100*v['stays']/v['windows']:.1f}% | {100*v['correct']/v['windows']:.1f}% | {v['correct_on_changes']:,}/{v['changes']:,} | {v['incorrect_on_stays']:,}/{v['stays']:,} |")
    lines+=['','[Per-game counts and checkpoint identity](expert-direction-persistence.json) · [Original held development results](held.json)']
    source.with_name('expert-direction-persistence.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    source.with_name('expert-direction-persistence.json').write_text(json.dumps(dict(comparison=digest(source),actor=actor['checkpoint'],games=games),indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
