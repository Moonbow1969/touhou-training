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
    p.add_argument('--four-step',action='store_true',help='Compare teacher-forced and self-propagated four-step training.')
    args=p.parse_args();d=args.data.resolve()
    model=d/('four-title-four-step-feedback' if args.four_step else 'four-title-direction-feedback')
    root=d/('four-step-feedback-original' if args.four_step else 'four-title-direction-feedback-native')
    root.mkdir(exist_ok=True)
    ref,alt=('teacher','propagated') if args.four_step else ('image','feedback')
    actors={a:model/a/'policy.ts' for a in (ref,alt)}
    start=1780000 if args.four_step else 1510000
    games={str(t):list(range(start+t*100+51,start+t*100+67)) for t in (10,17)}
    if args.four_step:
        freeze(root/'assigned-seeds.json',dict(eval_seeds=games,rank_rule='Odd Hard, even Lunatic. Eight matched pairs in each title and difficulty.',
            reference=ref,candidate=alt,training=digest(model/'protocol.json'),design='Run all seeds independent of intermediate fixed-screen results. Shared host and lane within each pair. Alternate arm order within rank. Start at Stage 1 with ordinary stock, play to terminal without Continue. All four controls come from each visual SFT. Report Stage 2, trajectory and measured inventory, retaining technical attempts.'))
        for arm in actors:
            while not (model/arm/'fit.json').is_file():time.sleep(10)
            if json.loads((model/arm/'fit.json').read_bytes())['actor']!=digest(actors[arm]):raise ValueError('Changed four-step actor')
    hashes={a:digest(f) for a,f in actors.items()}
    plan=dict(reference_arm=ref,candidate=alt,policies=hashes,eval_seeds=games,
              rank_by_seed={t:{str(i):2 if i%2 else 3 for i in seeds} for t,seeds in games.items()},
              sft_protocol_sha256=digest(model/'protocol.json'),
              design='Both human SFT variants use identical original images, architecture, training rows and action temperatures. Each native worker transmits the previous executed direction, initially neutral. The image control substitutes neutral inside its model. Same controller seed, build and lane in every pair. Original RNG is uncontrolled. Study starts from stock life and terminates at Game Over or ending, with no continues or automatic actions.',
              outcomes='Report Stage 2 and corresponding stocks when reached, terminal spares and payable Power or last-live Bombs per title and difficulty. Compare native last Stage 1 tick and uncertainty within each title and rank. The two titles were included in the human training and are independent runs of familiar titles.',
              scope='Eight paired controller seeds per difficulty per title. Per-title lane and execution-order assignments live beneath native10 and native17.')
    protocol=root/'camera-protocol.json'
    if args.four_step:
        plan.update(assignment=digest(root/'assigned-seeds.json'),design='Both feedback actors use their own preceding command, with neutral on reset. Teacher and propagated variants trained on the same human screens and four-command labels. Reports and checkpoints are bound to the training registration. Original RNG is uncontrolled within each matched pair.')
    else:plan['sft_evaluation_sha256']=digest(model/'report.json')
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
        '--parent',actors[ref],'--candidate',actors[alt],'--arm-names',ref,alt,'--out',out)
    run('summarize_four_title_pair','--root',root)
    run('plot_full_action_results','--folder',out,'--candidates',alt)
    for rank in ('Hard','Lunatic'):
        name='first-'+rank.lower()+'-native-credits.mp4' if args.four_step else rank.lower()+'-full-credits.mp4'
        run('render_cross_title_credits','--root',root,'--out',out/name,'--difficulty',rank,'--titles',10,17)


if __name__=='__main__':
    main()
