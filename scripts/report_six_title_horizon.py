"""Combine official stock-game endpoints for all twelve held game and difficulty cells."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze


def read(path):return json.loads(path.read_bytes())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--public',type=Path,required=True)
    a=parser.parse_args();root=a.data/'six-game-return-horizon';dest=a.public/root.name
    plan=read(root/'camera-protocol.json');arms=(plan['reference_arm'],plan['candidate'])
    controls=root/'evaluation/whole-game-controls.json';eight_file=root/'th08/evaluation/held-results.json'
    while not controls.is_file() or not eight_file.is_file():time.sleep(20)
    subprocess.run([sys.executable,'-m','scripts.report_th12_held_populations','--root',str(root)],check=True)
    (dest/'evaluation').mkdir(exist_ok=True)
    shutil.copy2(root/'evaluation/th12-physical-and-qualified.json',dest/'evaluation/th12-physical-and-qualified.json')
    five=read(controls);eight=read(eight_file)
    checkpoints={arm:read(root/arm/'policy/report.json')['policy_sha256'] for arm in arms}
    if eight['actor_binding']!=checkpoints or five['complete_credits']!=160 or len(eight['pairs'])!=16:
        raise ValueError('Incomplete registered population')
    receipts=read(root/'evaluation/credits.json')
    if digest(root/'evaluation/credits.json')!=five['credits_sha256']:raise ValueError('Unbound physical scores')
    keys={(v['title'],v['seed'],v['policy_sha256']) for v in receipts}
    if keys!={(int(t),s,h) for t,ss in plan['eval_seeds'].items() if t!='8' for s in ss for h in checkpoints.values()}:
        raise ValueError('Different frozen evaluation population')
    if {x['slot'] for x in eight['pairs']}!=set(plan['eval_seeds']['8']):raise ValueError('Missing TH08 matched slot')
    cells={}
    for title in (8,10,11,12,13,17):
        for rank in ('Hard','Lunatic'):
            key=f'TH{title:02} {rank}';cells[key]={}
            for arm in arms:
                if title==8:
                    subset=eight['paired'][rank];stage=subset['original_stage_evidence'][arm];metrics=subset['metrics']
                    v=dict(n=subset['n'],stage2=stage['confirmed_stage2'],unknown=stage['unknown'],spares=metrics['terminal_spares']['means'][arm],
                           resource='last playable Bombs',stock=metrics['last_playable_bombs']['means'][arm])
                else:
                    k=five['cells'][key][arm]
                    field,unit=('terminal_power','final Power') if title in (10,11) else ('last_playable_bombs','last playable Bombs') if title==17 else ('terminal_bombs','terminal Bombs')
                    v=dict(n=k['n'],stage2=k['stage2'],unknown=k['stage_unknown'],spares=k['terminal_spares']['mean'],resource=unit,stock=k[field]['mean'])
                if v['n']!=8:raise ValueError('Incomplete cell')
                cells[key][arm]=v
    result=dict(credits=192,paired_slots=96,sources={f.name:digest(f) for f in (controls,eight_file)},models=checkpoints,cells=cells)
    freeze(dest/'all-six-original-results.json',result)
    totals=[sum(v[arm]['stage2'] for v in cells.values()) for arm in arms];unknown=[sum(v[arm]['unknown'] for v in cells.values()) for arm in arms]
    lines=['# Six-game PPO on 192 new Hard and Lunatic original tests', '',
           f'The 96 paired starting allocations produced {totals[0]}/96 confirmed Stage 2 arrivals for the λ=0.985 actor and {totals[1]}/96 for the full-return actor. The respective counts of unresolved stages are {unknown[0]} and {unknown[1]}. All six titles supplied eight pairs at each difficulty. The two fits used the same 192 separate training credits and shared an initial policy, critic and out-of-fold baseline.', '',
           '| Game and difficulty | Stage 2, λ=.985 → full return | Unknown stage | Mean ending spare lives | Resource | Mean ending resource |', '|---|---:|---:|---:|---|---:|']
    for key,group in cells.items():
        x,y=[group[arm] for arm in arms]
        lines.append(f"| {key} | {x['stage2']}/8 → {y['stage2']}/8 | {x['unknown']} → {y['unknown']} | {x['spares']:.2f} → {y['spares']:.2f} | {x['resource']} | {x['stock']:.3f} → {y['stock']:.3f} |")
    lines+=['','TH10 and TH11 spend Power to cast. TH11 resets it after a fatal loss. TH08 and TH17 report Bomb stock from the last playable observation. Game RNG can differ within a pair. One TH08 Hard recorder lost host memory mid-game; the registered replacement rule preserved both original attempts and used a new paired seed. The TH12 physical denominator retains confirmed terminations with incomplete input coverage; the conditional comparison requires both sides to pass the control audit.', '',
            '[All twelve cells and source hashes](all-six-original-results.json) · [TH08 endpoints and color recordings](evaluation/TH08.md) · [Other five works, action traces and films](evaluation/README.md) · [TH12 coverage sensitivity](evaluation/th12-physical-and-qualified.json) · [Fitted policies](README.md)']
    (dest/'all-six-original-results.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    page=dest/'README.md';content=page.read_text(encoding='utf8');link='[All 192 independent original-game evaluation results](all-six-original-results.md)'
    if link not in content:page.write_text(content.rstrip()+'\n\n'+link+'\n',encoding='utf8')
    print('Confirmed six-title held outcomes',totals,'unresolved',unknown,flush=True)


if __name__=='__main__':main()
