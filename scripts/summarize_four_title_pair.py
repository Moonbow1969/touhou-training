"""Make a compact per-work, per-rank result table from qualified original credits."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    root=p.parse_args().root
    plan=json.loads((root/'camera-protocol.json').read_bytes())
    rows=json.loads((root/'evaluation/credits.json').read_bytes())
    results=json.loads((root/'evaluation/paired-by-difficulty.json').read_bytes())
    a,b=plan['reference_arm'],plan['candidate']
    names={10:'TH10',12:'TH12',13:'TH13',17:'TH17'}
    if len(rows)!=2*sum(map(len,plan['eval_seeds'].values())):raise ValueError('Incomplete paired results')
    counts={arm:sum(r['stage2_reached'] for r in rows if r['arm']==arm) for arm in (a,b)}
    out=['# Four-game Hard and Lunatic full-credit results','',
         f'Across {len(rows)//2} paired original-game starts, {a} reached Stage 2 on {counts[a]} credits and {b} on {counts[b]}. Each pair uses the same controller seed, physical host and original build. The score uses native stage evidence. The uncontrolled game RNG can differ within a pair.','',
         f'Values run from **{a}** to **{b}**. Stock is averaged across complete credits. TH10 spends Power to cast. TH17 shows stock on the last playable frame because the game replenishes Bombs after final death.','',
         '| Game | Difficulty | Pairs | Stage 2 | Spare lives at end | Power or Bombs | Effective casts, total |',
         '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for t,name in names.items():
        for rank in ('Hard','Lunatic'):
            cell=results[rank]['by_title'][str(t)]
            left,right=cell[a],cell['candidate']
            field='mean_terminal_power' if t==10 else 'mean_last_playable_bombs' if t==17 else 'mean_terminal_bombs'
            out.append(f"| {name} | {rank} | {left['n']} | {left['stage2']} → {right['stage2']} | {left['mean_terminal_spares']:.2f} → {right['mean_terminal_spares']:.2f} | {left[field]:.2f} → {right[field]:.2f} | {left['total_confirmed_bombs']} → {right['total_confirmed_bombs']} |")
    out.extend(['','[Per-seed outcomes and action traces](credits.json) · [Paired uncertainty by difficulty](paired-by-difficulty.json) · [Training and evaluation protocol](../camera-protocol.json)',''])
    (root/'evaluation/README.md').write_text('\n'.join(out),encoding='utf8')


if __name__=='__main__':
    main()
