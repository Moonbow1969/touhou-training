"""Prepare verified whole-game metrics and complete paired films for the named frozen comparison."""
import argparse
import json
import shutil
from pathlib import Path
import subprocess
import sys
import time

from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','public'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();pub=a.public.resolve()/root.name;out=root/'evaluation'
    plan=json.loads((root/'camera-protocol.json').read_bytes())
    titles=sorted(int(t) for t in plan['eval_seeds'] if t!='8')
    expected=sum(len(plan['eval_seeds'][str(t)])*2 for t in titles)
    def wait(path):
        while not path.is_file():time.sleep(15)
    scored=out/'credits.json';wait(out/(plan['candidate']+'.json'))
    if len(json.loads(scored.read_bytes()))!=expected:raise ValueError('Incomplete full-game score')
    def run(module,*argv):
        subprocess.run([sys.executable,'-m','scripts.'+module,*map(str,argv)],check=True)
    run('report_whole_game_controls','--root',root)
    film_kind='full' if root.name=='five-game-generation3' else 'native'
    for rank in ('Hard','Lunatic'):
        stem='first-'+rank.lower()+'-'+film_kind+'-credits'
        wait(out/(stem+'.json'))
        v=json.loads((out/(stem+'.json')).read_bytes())
        if v['sha256']!=digest(out/(stem+'.mp4')) or len(v['sources'])!=2*len(titles):raise ValueError('Video failed source validation')
    dest=pub/'evaluation';dest.mkdir(exist_ok=True,parents=True)
    for file in out.iterdir():
        duplicate_film=film_kind=='full' and file.name.startswith('first-') and '-native-credits' in file.name
        if file.is_file() and file.suffix in ('.png','.json','.mp4') and not duplicate_film:shutil.copy2(file,dest/file.name)
    for name in ('action-traces','endpoints'):
        shutil.copytree(out/name,dest/name,dirs_exist_ok=True)
    weights=json.loads((out/'whole-game-controls.json').read_bytes())
    summary=['# Paired full-stock original-game results','',
             f"Every title and difficulty supplied eight matched seeds for {plan['reference_arm']} and {plan['candidate']}. The full evaluation contains {expected} ordinary original-game starts. Both arms use the same build and host within each pair. In-game randomness is uncontrolled.",'',
             '| Original game | Stage 2, reference → candidate | Mean spare lives at end | Mean Bombs or Power at end | Confirmed casts, reference → candidate |',
             '|---|---:|---:|---:|---:|']
    for label,cell in weights['cells'].items():
        x,y=cell[plan['reference_arm']],cell[plan['candidate']]
        def resource(x):
            if x['last_playable_bombs']:return 'last playable Bomb',x['last_playable_bombs']['mean']
            if x['terminal_bombs']:return 'Bomb',x['terminal_bombs']['mean']
            if x['terminal_power']:return 'Power',x['terminal_power']['mean']
            return 'unavailable',None
        name,v=resource(x);other,u=resource(y)
        if name!=other:raise ValueError('Resource definitions differ across pair')
        metric=f'{name} {v:.3f} → {u:.3f}' if v is not None else name
        rates=f"{x['stage2']}/{x['n']} → {y['stage2']}/{y['n']}"
        if x['stage_unknown'] or y['stage_unknown']:rates+=f" (unknown {x['stage_unknown']} / {y['stage_unknown']})"
        summary.append(f"| {label} | {rates} | {x['terminal_spares']['mean']:.2f} → {y['terminal_spares']['mean']:.2f} | {metric} | {x['effective_casts']} → {y['effective_casts']} |")
    summary+=['',f'[Full-stock records](credits.json) · [Input and resource counts](whole-game-controls.json) · [Paired results by difficulty](paired-by-difficulty.json) · [Full Hard recording](first-hard-{film_kind}-credits.mp4) · [Full Lunatic recording](first-lunatic-{film_kind}-credits.mp4)','','![Observed original outcomes and physical inputs](whole-game-controls.png)']
    if (out/'README.md').is_file():
        summary=[(out/'README.md').read_text(encoding='utf8').rstrip(),'','[Native four-head input counts](whole-game-controls.json) · [Whole-game controls figure](whole-game-controls.png)']
    (dest/'README.md').write_text('\n'.join(summary)+'\n',encoding='utf8')
    freeze(out/'publication-summary.json',dict(verified_count=expected,raw_manifest=digest(scored),controls=digest(out/'whole-game-controls.json')))
    print('Mirrored verified ordinary games',root.name,expected,flush=True)

if __name__=='__main__':
    main()
