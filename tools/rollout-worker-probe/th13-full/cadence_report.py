"""Score the registered TH13 four-versus-eight-native-tick held comparison."""
import argparse,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
from paired_progress import compare
from summarize import wilson

def score(root,registration):
    plan=json.loads(registration.read_bytes())
    result=dict(preregistration_sha256=hashlib.sha256(registration.read_bytes()).hexdigest(),
                native_guard='350ms without any advancing native stage tick. Timing anchors are recorded immediately after X11 application and at release.',
                original_game_rng_controlled=False,groups=[])
    hosts=set()
    for ticks in (4,8):
        credits=[];armed=[]
        for seed in range(734001,734007):
            folder=root/f'held-sft{ticks}-{seed}';r=json.loads((folder/'receipt.json').read_bytes());hosts.add(r['host'])
            if r['policy_sha256']!=plan['policy_sha256'] or r['seed']!=seed or r['requested_native_ticks']!=ticks:raise ValueError(folder)
            if not r['terminal'] or r['continues'] or r['native_terminal_pause_state']!=1:raise ValueError(folder)
            for file,key in [('rollout.npz','sha256'),('cadence.npz','cadence_sha256')]:
                if hashlib.sha256((folder/file).read_bytes()).hexdigest()!=r[key]:raise ValueError(folder/file)
            with np.load(folder/'cadence.npz') as c, np.load(folder/'rollout.npz') as z:
                regular=c['stop_reason']==0;g=c['native_tick_at_release']-c['native_tick_at_action'];wall=c['hold_us']
                if not np.all(g[regular]>=ticks) or len(g)!=r['steps']:raise ValueError(folder)
                armed.append(np.column_stack([g,c['stop_reason'],wall]))
                native_tick_delta=int(c['native_tick_at_release'][-1]-c['native_tick_at_release'][0])
                active_seconds=float(z['seconds'][-1]-z['seconds'][0])
            credits.append({k:r[k] for k in ['seed','start_utc','finish_utc','steps','stage2','stage2_entry_spares','stage2_entry_bombs','ranking_active_lives','ranking_bombs','native_ranking_tick','ranking_display_score','last_playable_bombs','effective_bombs','wall_seconds','native_key_match','advancing_stops','exact_target_stops','overshoot_stops','guard_stops','transition_stops','fatal_stops','median_hold_ms','sha256','cadence_sha256']})
            credits[-1].update(native_tick_delta_first_last_decision=native_tick_delta,elapsed_first_last_decision_seconds=active_seconds,
                               observed_native_ticks_per_second=native_tick_delta/active_seconds)
        timing=np.concatenate(armed);gap,why,us=timing.T;normal=why==0;guard=why==1;arrivals=sum(r['stage2'] for r in credits)
        result['groups'].append(dict(ticks=ticks,n=len(credits),stage2=arrivals,stage2_wilson_95=wilson(arrivals,len(credits)),decision_steps=len(gap),mean_terminal_active_lives=float(np.mean([r['ranking_active_lives'] for r in credits])),mean_terminal_bombs=float(np.mean([r['ranking_bombs'] for r in credits])),mean_effective_bombs=float(np.mean([r['effective_bombs'] for r in credits])),mean_native_ranking_tick=float(np.mean([r['native_ranking_tick'] for r in credits])),median_native_ranking_tick=float(np.median([r['native_ranking_tick'] for r in credits])),mean_policy_wall_seconds=float(np.mean([r['wall_seconds'] for r in credits])),regular_advancing_stops=int(normal.sum()),exact_requested_ticks=int((normal&(gap==ticks)).sum()),exact_fraction_of_advancing=float((gap[normal]==ticks).mean()),advancing_gaps_min_median_p95_max=np.percentile(gap[normal],[0,50,95,100]).tolist(),elapsed_hold_ms_median=float(np.median(us)/1000),no_progress_guard_stops=int(guard.sum()),guard_stopped_with_zero_native_ticks=int((guard&(gap==0)).sum()),menu_or_stage_transition_stops=int((why==2).sum()),native_end_life_stops=int((why==3).sum()),rows=credits))
        result['groups'][-1]['median_observed_native_ticks_per_second']=float(np.median([r['observed_native_ticks_per_second'] for r in credits]))
    if len(hosts)!=1:raise ValueError('Cross-host cadence confound')
    result['host']=next(iter(hosts));result['paired']=compare(root,['sft4','sft8'],range(734001,734007),261125)['pairs'][0]
    clock=root/'four-lane-clock.json'
    if clock.is_file():result['whole_pool_throughput']=json.loads(clock.read_bytes())
    return result

def figure(data,path):
    import matplotlib.pyplot as plt
    f,axes=plt.subplots(1,3,figsize=(10.7,4))
    groups=data['groups'];t=[4,8];palette=['#4b6aaf','#13816e'];x=np.arange(2)
    for i in range(6):
        axes[0].plot(x,[g['rows'][i]['native_ranking_tick'] for g in groups],color='#b6bec3',lw=.9)
        axes[1].plot(x,[g['rows'][i]['last_playable_bombs'] for g in groups],color='#b6bec3',lw=.9)
    for k,g in enumerate(groups):
        for a,key in [(axes[0],'native_ranking_tick'),(axes[1],'last_playable_bombs')]:
            y=[r[key] for r in g['rows']];a.scatter([k]*6,y,c=palette[k],s=30,zorder=3);a.plot([k-.13,k+.13],[np.mean(y)]*2,c='#15242a',lw=2.2,zorder=4)
        axes[2].bar(k,g['elapsed_hold_ms_median'],width=.5,color=palette[k])
        axes[2].text(k,g['elapsed_hold_ms_median']+3,f"{g['elapsed_hold_ms_median']:.0f} ms",ha='center')
    for a in axes:
        a.set_xticks(x,['4 ticks','8 ticks']);a.set_xlim(-.48,1.48);a.spines[['top','right']].set_visible(False);a.grid(axis='y',alpha=.17)
    axes[0].set_title('Stage 1 ranking tick');axes[1].set_title('Bombs before final life loss');axes[1].set_ylim(-.18,3.1);axes[1].set_yticks([0,1,2,3]);axes[2].set_title('Median action hold');axes[2].set_ylabel('Wall milliseconds');axes[2].set_ylim(0,max(g['elapsed_hold_ms_median'] for g in groups)*1.25)
    f.suptitle('TH13 original trial  ﾂｷ  same frozen human SFT  ﾂｷ  Normal Reimu',fontsize=15)
    f.text(.5,.02,'Six matched controller seeds per arm, native rankings, no Continues. Lines pair seeds; black marks show means.',ha='center',fontsize=9)
    f.tight_layout(rect=(0,.06,1,.93));f.savefig(path,dpi=145);plt.close(f)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('registration',type=Path);p.add_argument('--release',type=Path);args=p.parse_args()
    result=score(args.root,args.registration);path=args.root/'four-vs-eight-native-ticks.json';path.write_text(json.dumps(result,indent=2)+'\n');figure(result,path.with_suffix('.png'))
    if args.release:
        args.release.mkdir(parents=True,exist_ok=True)
        for f in [path,path.with_suffix('.png'),args.root/'host-qualification.json',args.root/'qualification.png',args.root/'cadence-preregistration.json',args.root/'cadence-host-amendment.json']:shutil.copyfile(f,args.release/f.name)
        with zipfile.ZipFile(args.release/'cadence-native-traces.zip','w') as z:
            for arm in (4,8):
                for seed in range(734001,734007):
                    folder=args.root/f'held-sft{arm}-{seed}'
                    for n in ['receipt.json','rollout.npz','cadence.npz','events.tsv','actor.log','final-stock.bin']:
                        z.write(folder/n,f'{folder.name}/{n}',compress_type=zipfile.ZIP_STORED if n.endswith('.npz') else zipfile.ZIP_DEFLATED)
