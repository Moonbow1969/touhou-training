"""Track held human behavior at fixed images across linked SFT and multigame PPO checkpoints."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from scripts.checkpoint_storage import digest
from scripts.train_causal_replay_history import paths
from scripts.train_five_original_games import freeze


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True,type=Path)
    d=p.parse_args().data.resolve();study=d/'six-game-human-retention';study.mkdir(exist_ok=True)
    sft=d/'specialist-vs-shared/merged-equal';g1=d/'five-game-online/policy';g2=d/'five-game-generation2/policy'
    dirs={'SFT':sft,'five-G1':g1,'five-G2':g2,'five-G3':d/'five-game-generation3/policy',
          'five-crossfit':d/'five-game-crossfit-advantages/policy','six-G1':d/'six-game-online/joint-six/policy'}
    repeated=json.loads((d/'specialist-vs-shared/protocol.json').read_bytes())['held']
    freshdir=d/'human-retail17-wave2-held'
    plan=dict(actors={k:digest(v/'policy.ts') for k,v in dirs.items()},repeated_development=repeated,
        fresh_roster=digest(freshdir/'frozen-source-roster.json'),
        purpose='Test whether the movement distributions after online updates still predict expert actions on the same original human screens. Report exact direction prediction and timing-shift control, separately by title and player. Link each online actor to the original SFT by both behavior and parameter hashes. This measures imitation on recorded expert routes. Ordinary-stock performance is reported separately. The four new TH17 players stay out of all optimizer inputs.')
    freeze(study/'protocol.json',plan)
    lookup=paths(d)
    for cohort in ('previous-development','new-th17-players'):
        if cohort=='previous-development':
            stages=[lookup[h] for group in repeated.values() for h in group]
        else:
            final=freshdir/'held-original-synchronized-frozen.json'
            while not final.exists():time.sleep(15)
            qualified=json.loads(final.read_bytes())['qualified']
            stages=[d/r['path'] for r in qualified]
            if any(r['split']!='held' or digest(f)!=r['archive_sha256'] for f,r in zip(stages,qualified)):
                raise ValueError('Unverified independent held list')
        dest=study/cohort
        if not (dest/'held.json').exists():
            subprocess.run([sys.executable,'-m','scripts.compare_human_camera_policies','--models',*dirs.values(),
                '--model-names',*dirs,'--sft-parent',sft,'--ppo-ancestors',g1,g2,'--held',*stages,'--out',dest],check=True)
        raw=json.loads((dest/'held.json').read_bytes());by_title={}
        for arm,score in raw.items():
            if score['checkpoint']!=plan['actors'][arm]:raise ValueError('Unbound actor')
            sums={}
            for f in stages:
                meta=json.loads(f.with_suffix('.json').read_bytes());t=meta['title'];record=score['replays'][f.name]
                for kind in ('aligned','shifted'):
                    for mode in ('all','change','start','turn'):
                        x=record[kind][mode]
                        acc=sums.setdefault((t,kind,mode),[0,0])
                        acc[0]+=x['n'];acc[1]+=x['correct']
            by_title[arm]={}
            for (t,kind,mode),(n,hits) in sums.items():
                by_title[arm].setdefault(t,{}).setdefault(kind,{})[mode]=dict(n=n,correct=hits,rate=hits/n if n else None)
        freeze(dest/'by-title.json',dict(held_sources=[digest(f) for f in stages],actors=by_title))
        print('Sealed human trajectory metrics',cohort,len(stages),flush=True)


if __name__=='__main__':
    main()
