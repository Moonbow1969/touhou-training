"""Measure how three same-parent PPO fits change imitation on fixed held human observations."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from scripts.checkpoint_storage import digest
from scripts.report_shared_title_training import scores


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    data=parser.parse_args().data.resolve()
    def read(path):
        return json.loads(path.read_bytes())
    parent=data/'specialist-vs-shared/merged-equal'
    models=[parent,*[data/name/'policy' for name in ('four-game-merged-online','th11-original-ppo','five-game-online')]]
    names=['merged-sft','four-ppo','th11-ppo','five-ppo']
    directory=data/'five-game-online/held-human-movement'
    split=read(data/'ranked-multigame-sft/design.json')['base_split']
    paths={r['sha256']:data/r['file'] for r in read(data/split)['split']['held']}
    paths.update({r['archive_sha256']:data/r['path'] for r in read(data/'human-retail17/held-qualified-02-stages-frozen.json')['qualified']})
    registrations=read(data/'specialist-vs-shared/protocol.json')['held']
    results={}
    for title,hashes in registrations.items():
        selected=[paths[h] for h in hashes]
        if any(digest(p)!=h for p,h in zip(selected,hashes)):
            raise ValueError('Changed human replay')
        dest=directory/title
        subprocess.run([sys.executable,'-m','scripts.compare_human_camera_policies','--models',*models,'--model-names',*names,
                        '--sft-parent',parent,'--held',*selected,'--out',dest],check=True)
        results[title]={}
        for name,arm in read(dest/'held.json').items():
            rows=[r['aligned'] for r in arm['replays'].values()]
            results[title][name]={c:scores(rows,c) for c in ('all','change','start','turn')}
        print(title,json.dumps({a:round(v['change']['accuracy'],4) for a,v in results[title].items()}),flush=True)
    report=dict(interpretation='Repeated development players. Differences are paired on the same camera, previous image, native input and label. Online PPO fits all share the identical human SFT parent. No TH11 human pixels have qualified for this split. Metrics here describe imitation on the human state distribution. Original game survival is measured independently.',
                checkpoints={name:digest(directory/'policy.ts') for name,directory in zip(names,models)},
                held_camera_hashes=registrations,by_title=results)
    (directory/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    main()
