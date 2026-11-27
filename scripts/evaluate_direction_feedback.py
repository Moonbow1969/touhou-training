"""Register and score independent original-game credits for both movement heads."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.checkpoint_storage import digest
from scripts.score_original_camera_run import records
from scripts.train_five_original_games import freeze


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    d=p.parse_args().data.resolve()
    model=d/'four-title-direction-feedback'
    root=d/'four-title-direction-feedback-native'
    root.mkdir(exist_ok=True)
    actors={a:model/a/'policy.ts' for a in ('image','feedback')}
    hashes={a:digest(f) for a,f in actors.items()}
    games={str(t):list(range(1510000+t*100+51,1510000+t*100+67)) for t in (10,17)}
    plan=dict(reference_arm='image',candidate='feedback',policies=hashes,eval_seeds=games,
              rank_by_seed={t:{str(i):2 if i%2 else 3 for i in seeds} for t,seeds in games.items()},
              sft_protocol_sha256=digest(model/'protocol.json'),sft_evaluation_sha256=digest(model/'report.json'),
              design='Both human SFT variants use identical original images, architecture, training rows and action temperatures. Each native worker transmits the previous executed direction, initially neutral. The image control substitutes neutral inside its model. Same controller seed, build and lane in every pair. Original RNG is uncontrolled. Study starts from stock life and terminates at Game Over or ending, with no continues or automatic actions.',
              outcomes='Report Stage 2 and corresponding stocks when reached, terminal spares and payable Power or last-live Bombs per title and difficulty. Compare native last Stage 1 tick and uncertainty within each title and rank. The two titles were included in the human training and are independent runs of familiar titles.',
              scope='Eight paired controller seeds per difficulty per title. Per-title lane and execution-order assignments live beneath native10 and native17.')
    protocol=root/'camera-protocol.json'
    freeze(protocol,plan)
    completed=set()
    while len(completed)<2:
        for title in (10,17):
            if title in completed:
                continue
            wanted={(s,h) for s in games[str(title)] for h in hashes.values()}
            found=set()
            for t,f in records(root,True,True):
                if t!=title:
                    continue
                try:
                    r=json.loads(f.read_bytes())
                except (json.JSONDecodeError,PermissionError):
                    continue
                key=(r.get('seed'),r.get('policy_sha256'))
                if key not in wanted:
                    continue
                if key in found:
                    raise ValueError('Duplicate physical receipt')
                if title==17:
                    audited=f.parent/'independent-policy-audit.json'
                    if not audited.is_file():
                        continue
                    a=json.loads(audited.read_bytes())
                    if (a.get('verified_previous_move_rows')!=r['actions'] or a['policy_sha256']!=key[1]
                        or a['trajectory_sha256']!=r['trajectory_sha256']):
                        raise ValueError('Unverified full-credit history transfer')
                found.add(key)
            if wanted==found:
                print('Physical matched pairs ready',title,flush=True)
                completed.add(title)
        if len(completed)<2:
            time.sleep(20)
    def run(module,*args):
        subprocess.run([sys.executable,'-m','scripts.'+module,*map(str,args)],check=True)
    out=root/'evaluation'
    run('score_original_camera_run','--root',root,'--round2','--protocol',protocol,
        '--parent',actors['image'],'--candidate',actors['feedback'],'--arm-names','image','feedback','--out',out)
    run('summarize_four_title_pair','--root',root)
    run('plot_full_action_results','--folder',out,'--candidates','feedback')
    for rank in ('Hard','Lunatic'):
        run('render_cross_title_credits','--root',root,'--out',out/(rank.lower()+'-full-credits.mp4'),'--difficulty',rank)


if __name__=='__main__':
    main()
