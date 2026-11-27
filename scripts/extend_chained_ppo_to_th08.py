"""Compare five and six original titles using the same frozen actor, critic and old-game experiences."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze


def read(path):
    return json.loads(path.read_bytes())


def wait(path):
    while not path.is_file():
        time.sleep(15)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--public',type=Path,required=True)
    args=parser.parse_args()
    data=args.data.resolve()
    source=data/'reviewed-five-human-online/policy'
    sibling=data/'chained-five-game-ppo'
    root=data/'chained-six-game-ppo'
    root.mkdir(exist_ok=True)
    public=args.public.resolve()/root.name
    public.mkdir(exist_ok=True)
    prior=read(sibling/'camera-protocol.json')
    actor=digest(source/'policy.ts')
    if actor!=prior['behavior_sha256'] or digest(source/'camera-ppo.pt')!=prior['parent_actor_critic_sha256']:
        raise ValueError('Both fits must restore the same deployed actor and critic')
    new=list(range(1890801,1890817))
    held=list(range(1890851,1890883))
    plan=dict(name='Add fresh TH08 training to the shared five-game second PPO generation',
        reference_arm='five-ppo2',candidate='six-ppo2',behavior_sha256=actor,parent_actor_critic_sha256=prior['parent_actor_critic_sha256'],
        parent_five_protocol_sha256=digest(sibling/'camera-protocol.json'),
        games={**prior['games'],'TH08':'Original publisher trial 0.03'},train_seeds={**prior['train_seeds'],'8':new},
        eval_seeds={'8':held},rank_by_seed={**prior['rank_by_seed'],'8':{str(s):2 if s%2 else 3 for s in new+held}},
        controls=prior['controls'],ppo=prior['ppo'],learning_rates=prior['learning_rates'],balanced_title_rank=True,
        training='Use precisely the same selected five-game source files as the five-title sibling and append sixteen fresh full-stock TH08 training games, eight per difficulty, from the common parent. No previous evaluation is eligible. Each of twelve title/rank strata receives equal weight. Both fits use the same number of optimizer minibatches per attempted epoch and the same KL acceptance rule. Adding data changes the per-game share, per-step batch size and sampling order. Report any difference in accepted epochs.',
        evaluation='TH08 contributes sixteen new pairs per difficulty. Compare the five-title and six-title siblings on the same physical host, lane and native build within each paired seed. Use Stage 2 as the primary endpoint, then first life loss, time to Game Over, terminal spare lives and last playable Bombs. Original-game randomness is uncontrolled. This comparison evaluates adaptation to TH08. It does not measure retention on the other five games.')
    freeze(root/'camera-protocol.json',plan)
    shutil.copy2(root/'camera-protocol.json',public)
    print('Registered shared-parent TH08 expansion',digest(root/'camera-protocol.json'),flush=True)
    native=root/'qualified-train.json'
    wait(native)
    wait(sibling/'policy/report.json')
    five=sibling/'corrected/credits.json'
    report=read(sibling/'policy/report.json')
    if digest(five)!=report['manifest_sha256'] or report['native_behavior_sha256']!=actor or report['policy_sha256']!=digest(sibling/'policy/policy.ts'):
        raise ValueError('Invalid fitted five-title comparator')
    effective=read(sibling/'fit-protocol.json')
    expected={(int(t),s) for t,seeds in effective['train_seeds'].items() for s in seeds}|{(8,s) for s in new}
    old=read(five)
    extra=read(native)['rows']
    batch=root/'corrected'
    batch.mkdir(exist_ok=True)
    rows=[]
    for source_dir,group in ((five.parent,old),(root,extra)):
        for row in group:
            title,seed=row['title'],row['seed']
            raw=(source_dir/row['file']).resolve()
            raw.relative_to(source_dir)
            if (title,seed) not in expected or row['policy_sha256']!=actor or row['role']!='train' or row['difficulty']!=plan['rank_by_seed'][str(title)][str(seed)]:
                raise ValueError('Unregistered training input')
            if not row['difficulty_evidence_sha256'] or digest(raw)!=row['sha256']:
                raise ValueError('Unverified original training input')
            target=batch/f'th{title}-{seed}.npz'
            if not target.exists():os.link(raw,target)
            if digest(target)!=row['sha256']:raise ValueError('Mismatched linked input')
            rows.append(dict(row,file=target.name))
    if len(rows)!=len(expected) or {(r['title'],r['seed']) for r in rows}!=expected or len(extra)!=16:
        raise ValueError('Missing or duplicate training game')
    manifest=batch/'credits.json'
    freeze(manifest,rows)
    chunks=max(1,round(report['steps']/144))
    if any(item.get('optimizer_steps',chunks)!=chunks for item in report['history']):
        raise ValueError('Sibling update count disagrees with the training recipe')
    protocol=root/'fit-protocol.json'
    freeze(protocol,dict(plan,registered_protocol_sha256=digest(root/'camera-protocol.json'),
                         train_seeds={**effective['train_seeds'],'8':new},five_source_manifest=digest(five),th08_manifest=digest(native),
                         five_result_sha256=digest(sibling/'policy/report.json'),matched_optimizer_minibatches=chunks))
    if not (root/'policy/report.json').exists():
        subprocess.run([sys.executable,'-u','-m','scripts.train_full_resolution_camera_ppo','--manifest',str(manifest),
                        '--protocol',str(protocol),'--behavior',str(source/'policy.ts'),'--parent-actor-critic',str(source/'camera-ppo.pt'),
                        '--out',str(root/'policy')],check=True)
    result=read(root/'policy/report.json')
    if (result['policy_sha256']!=digest(root/'policy/policy.ts') or result['protocol_sha256']!=digest(protocol)
            or result['parent_actor_critic_sha256']!=plan['parent_actor_critic_sha256'] or len(result['equal_weight_strata'])!=12
            or any(item['optimizer_steps']!=chunks for item in result['history'])):
        raise ValueError('Unverified six-title fit')
    for path in (manifest,protocol,root/'policy/report.json',root/'policy/policy.ts',root/'policy/camera-ppo.pt'):
        dest=public/path.relative_to(root)
        dest.parent.mkdir(exist_ok=True,parents=True)
        shutil.copy2(path,dest)
    phases=[i for i in result['history'] if i['accepted']]
    (public/'README.md').write_text('\n'.join([
        '# Six-title full-action PPO from the five-title human policy','',
        f"The learner restored the same trained actor and critic as the five-title comparator. It consumed all {len(old)} selected games from that comparator and 16 fresh TH08 games, with eight Hard and eight Lunatic TH08 starts. The shared six-title checkpoint learned from {result['steps']:,} playable decisions across twelve game/difficulty combinations.",
        '',f"Both recipes attempted {chunks} optimizer steps per epoch. The six-title fit accepted {len(phases)} epochs. Adding TH08 changes the fraction of the batch allocated to each previously trained work. The separate 64-game TH08 evaluation uses prespecified controller seeds and fixed policy weights.",
        '', '[Source allocation](camera-protocol.json) · [Exact selected seeds](fit-protocol.json) · [Full optimizer log](policy/report.json) · [Deployable policy](policy/policy.ts) · [Existing five-title input archives](../chained-five-game-ppo/README.md)',
        '']),encoding='utf8')
    print('Completed six-title fit',len(result['credits']),result['steps'],result['policy_sha256'],flush=True)


if __name__=='__main__':main()
