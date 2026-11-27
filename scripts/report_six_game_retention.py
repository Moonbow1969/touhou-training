"""Combine all 96 original full credits in the five-title versus six-title checkpoint comparison."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.checkpoint_storage import digest

def read(f):return json.loads(f.read_bytes())


def main():
    arg=argparse.ArgumentParser(description=__doc__);arg.add_argument('--data',type=Path,required=True);arg.add_argument('--public',type=Path,required=True)
    a=arg.parse_args();root=a.data.resolve()/'six-game-retention';dest=a.public.resolve()/'six-game-retention';dest.mkdir(exist_ok=True)
    files=[root/'evaluation/credits.json',root/'th08/evaluation/held-results.json'];others=read(files[0]);eight=read(files[1]);plan=read(root/'camera-protocol.json')
    arms=[plan['reference_arm'],plan['candidate']]
    if len(others)!=64 or len(eight['pairs'])!=16 or list(eight['actor_binding'])!=arms:raise ValueError('Missing registered held games')
    records={}
    for r in others:
        key=r['title'],r['difficulty'],r['seed'],r['arm']
        if key in records or r['policy_sha256']!=eight['actor_binding'][r['arm']]:raise ValueError('Changed original actor')
        resource=r.get('last_playable_bombs') if r['title']==17 else r['terminal_power']
        records[key]=dict(stage2=r['stage2_reached'],spares=r['terminal_spares'],stock=resource,progress=r['latest_observed_stage1_native_tick'],host=r['host'],build=r['game_sha256'])
    for pair in eight['pairs']:
        for arm,r in pair['rows'].items():
            audit=r['audit'];rank=2 if pair['rank']=='Hard' else 3
            records[8,rank,pair['slot'],arm]=dict(stage2=r['stage2'],spares=audit['terminal_spares'],stock=audit['last_playable_bombs'],progress=audit['observed_duration_s'],host='Gamer',build='TH08 0.03')
    cells={};rng=np.random.default_rng(169111);fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
    page=['# Five-title and six-title PPO on 96 ordinary original credits','',
          'Both policies start from the same five-title generation-two actor and use the same 160 original games on TH10, 11, 12, 13 and 17. The six-title fit also used 32 TH08 games, with equal weight per game and difficulty. The comparison evaluates TH08 acquisition and retention on TH10 and TH17. Each difficulty supplied eight independent held controller seeds per title. The two policies shared a build and host within every pair. Native game randomness was uncontrolled.','',
          '| Game | Difficulty | Stage 2, five → six | Mean spare lives | Last playable Bombs or final Power | Mean progress, five → six | Paired change [95% interval] |','|---|---|---:|---:|---:|---:|---:|']
    for ax,t in zip(axes,(8,10,17)):
        for r in (2,3):
            rank={2:'Hard',3:'Lunatic'}[r];seeds=[s for s in plan['eval_seeds'][str(t)] if plan['rank_by_seed'][str(t)][str(s)]==r]
            pairs=[[records[t,r,s,h] for h in arms] for s in seeds]
            if len(pairs)!=8 or any(x['host']!=y['host'] or x['build']!=y['build'] for x,y in pairs):raise ValueError('Unmatched physical pair')
            if any(v['stage2'] is None or v['stock'] is None for pair in pairs for v in pair):raise ValueError('Unresolved stock or stage outcome')
            def values(key):return np.asarray([[v[key] for v in pair] for pair in pairs])
            delta=values('progress')[:,1]-values('progress')[:,0]
            # Resample within the physical host assignment to retain its original weight.
            hosts=np.array([pair[0]['host'] for pair in pairs])
            groups=[delta[hosts==h] for h in sorted(set(hosts))]
            draws=np.column_stack([rng.choice(x,(10000,len(x))) for x in groups]).mean(1)
            low,high=np.quantile(draws,[.025,.975]);mean=delta.mean();m=values('progress').mean(0);z=values('stage2').sum(0)
            b=values('stock').mean(0);l=values('spares').mean(0)
            label='s' if t==8 else 'ticks';stock='Power' if t==10 else 'Bombs'
            page.append(f'| TH{t:02} | {rank} | {z[0]}/8 → {z[1]}/8 | {l[0]:.1f} → {l[1]:.1f} | {stock} {b[0]:.3f} → {b[1]:.3f} | {m[0]:,.1f} → {m[1]:,.1f} {label} | {mean:+,.1f} [{low:+,.1f}, {high:+,.1f}] |')
            cells[f'TH{t:02} {rank}']=dict(seeds=seeds,arms=arms,stage2=z.tolist(),spares=l.tolist(),stock=b.tolist(),progress=m.tolist(),delta=mean,within_host_paired_bootstrap95=[low,high],paired=delta.tolist())
            ax.scatter(delta,np.full(8,3-r)+np.linspace(-.12,.12,8),s=16,color='#a7a7a7',zorder=2)
            ax.plot([low,high],[3-r,3-r],color='#086c85',lw=3);ax.scatter(mean,3-r,color='#086c85',s=38,zorder=3)
        ax.set_title(f'TH{t:02}');ax.axvline(0,color='#444',lw=.9);ax.set_yticks([0,1],['Lunatic','Hard']);ax.set_ylim(-.5,1.5);ax.set_xlabel('Original elapsed seconds' if t==8 else 'Stage 1 native ticks')
    fig.suptitle('Change in full-credit progress after adding TH08 training');fig.savefig(dest/'three-game-progress.png',dpi=160);plt.close(fig)
    page+=['','Both arms exhausted stock on Stage 1 in all 48 games. On TH08 Hard, adding TH08 training corresponded to a shorter mean credit in this cohort. The intervals describe paired original trials and do not adjust for multiple endpoints. TH08 measures elapsed play to the first Continue. The other two games measure the last observed playable Stage 1 tick. The TH17 score screen resets Bombs, so the table uses the last playable inventory.','',
        '![Paired progress and individual seeds](three-game-progress.png)','','[All matched results](three-game-results.json) · [TH08 stock records and original colour recordings](evaluation/TH08.md) · [TH10 and TH17 stock records and videos](evaluation/README.md) · [Initial training gradient alignment](initial-gradients/report.json)']
    (dest/'three-game-results.json').write_text(json.dumps(dict(source_sha256=[digest(f) for f in files],credits=len(records),cells=cells),indent=2)+'\n',encoding='utf8')
    (dest/'README.md').write_text('\n'.join(page)+'\n',encoding='utf8')


if __name__=='__main__':
    main()
