"""Measure stock TH13 Easy actor throughput at one, two and four concurrent originals on Accelerator."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from scripts.collect_th10_full_action import save
from scripts.run_th10_camera_cohort import digest
from scripts.run_th13_accelerator_camera import CameraPool,NATIVE


class MeteredPool(CameraPool):
    def __init__(self,o,monitor):
        super().__init__(o);self.monitor=monitor;self.cost={};self.downloaded=0;self.uploaded=0

    def remote(self,command,data=None,timeout=45):
        result=super().remote(command,data,timeout);self.downloaded+=len(result);self.uploaded+=len(data or b'')
        return result

    def boot(self,ident,ticks):
        start=time.perf_counter();name,base=super().boot(ident,ticks)
        self.cost.update(native_container=name,qualified_boot_s=time.perf_counter()-start)
        self.base=base;return name,base

    def observe_inference_cost(self,pid,lib):
        # Native wine and the host's separate PyTorch interpreter have separate cgroups.
        body=f"test -r /proc/{pid}/stat && tr '\\0' '\\n' </proc/{pid}/cmdline | grep -Fxq {shlex.quote(lib)} && cat /proc/{pid}/stat && getconf CLK_TCK"
        stat,hz=self.remote(body).decode().splitlines()
        values=stat[stat.rfind(')')+2:].split()
        self.cost['inference_cpu_s']=(int(values[11])+int(values[12]))/int(hz)

    def release(self,name):
        try:
            if getattr(self,'base',None) and self.cost.get('native_container')==name:
                raw=self.remote(shlex.join(['python3',self.monitor,'cgroup',name,self.base]))
                self.cost.update(json.loads(raw))
        finally:super().release(name)


def run(root,workers):
    study=root/'engineering/th13-accelerator-concurrency';study.mkdir(parents=True,exist_ok=True)
    s=root/'five-game-generation3'
    ids=list(range(1611301,1611309));model=root/'specialist-vs-shared/merged-equal/policy.ts'
    sha=digest(model);assert sha=='1f7f1a6b73fee215c0142a9a27be8d1523d38baa9ad99fba3d37d4839a792b3a'
    waves=[]
    for ix,order in enumerate(((4,1,2),(2,1,4))):
        group=ids[4*ix:4*ix+4]
        for workers_count in order:
            for j in range(0,4,workers_count):
                waves.append(dict(block=ix+1,concurrency=workers_count,seeds=group[j:j+workers_count]))
    protocol=dict(title=13,difficulty=0,opening_spares=2,opening_bombs=2,host='accelerator',character='Reimu',
        game_sha256='09818929cceea76ff4c65630e649bbccdb85d92b2e28bb4dfe6526749e6f4fb6',policy_sha256=sha,
        role='engineering',control='Unassisted original story, all four policy heads; four native ticks per decision.',
        waves=waves,unit='Eight controller seeds repeated at each concurrency, every credit fresh and independently subject to the original RNG. Block 1 order 4,1,2; block 2 order 2,1,4.',
        primary='Total verified decisions / sum of active whole-host acquisition seconds across all waves at this level; report native ticks / same denominator. Wave wall starts at parallel dispatch and ends after all native receipts and transfer complete. Staging and formal-study parking time excluded and reported separately.',
        quality='Every credit requires original Easy 0, full 2/2 opening, original Game Over, no Continue, physical command acknowledgement, actual action consumption. Report action-clock quantiles, confirmed ticks per decision, original endpoint and model identity.',
        resources='Linux host CPU jiffies and available RAM/tmpfs before and after each wave; each Wine cgroup CPU and I/O at exit plus PyTorch inference CPU and capture-directory bytes. Monitor nonbenchmark games, avoid formal 159/160 contention.',
        cost_provenance='One-time code/model upload outside measured waves. All individual games start fresh Wine/Xvfb and inference process.')
    reg=study/'protocol.json'
    if reg.exists():assert json.loads(reg.read_bytes())==protocol
    else:save(reg,protocol)
    conf=json.loads(workers.read_bytes())['accelerator']['command'][:-1]
    profile=json.loads((root/'engineering/th13-accelerator/deployment.json').read_bytes())
    hostroot=profile['private_remote_root'];runtime='/tmp/th13-bench-161';launcher=runtime+'/launch'
    def rem(cmd,data=None):
        return subprocess.run([*conf,cmd],input=data,check=True,capture_output=True,timeout=90).stdout
    assert rem("docker image inspect --format '{{.Id}}' "+profile['local_image']).decode().strip()==profile['local_image_id']
    rem('mkdir -p '+launcher)
    src=NATIVE.parent/'th13-pixels'
    for name in ('launch.sh','run-th13.sh','pixel-capture.c','x11_pixels.h'):
        payload=(src/name).read_bytes().replace(b'\r\n',b'\n')
        if name=='launch.sh':
            token=b'            if [ "$screen" = difficulty ] && [ "$TH13_RANK" -gt 1 ]; then'
            assert payload.count(token)==1
            payload=payload.replace(b'case "$rank" in 1|2|3)',b'case "$rank" in 0|1|2|3)').replace(token,
                b'            if [ "$screen" = difficulty ] && [ "$TH13_RANK" = 0 ]; then xdotool keydown Up; sleep .15; xdotool keyup Up; sleep .4; fi\n'+token)
        rem('cat > '+launcher+'/'+name,data=payload)
    monitor=runtime+'/host_cost.py';rem('cat > '+monitor,data=Path(__file__).with_name('measure_th13_actor_host.py').read_bytes().replace(b'\r\n',b'\n'))
    py=rem('v="$HOME/.local/share/toho-ml/distributed-ppo/venv/bin/python"; "$v" -c "import torch" && echo "$v"').decode().strip()
    known=next(x.split('=',1)[1] for x in conf if x.startswith('UserKnownHostsFile='))
    local=study/'native13';local.mkdir(exist_ok=True)
    def options(level):
        return SimpleNamespace(host=conf[-1],host_label='accelerator',known_hosts=Path(known),remote_root=runtime,
            prior_bridge=hostroot+'/toho-th13-cadence-261125/bridge',assets=hostroot+'/toho-th13-261125/assets/app',launcher=launcher+'/launch.sh',
            image=profile['local_image'],python_executable=py,checkpoint=model,output=local,role='engineering',arm=f'capacity{level}',difficulty=0,verified_easy_launcher=True)
    setup=time.perf_counter();first=MeteredPool(options(1),monitor);first.prepare()
    save(study/'setup.json',dict(setup_seconds=time.perf_counter()-setup,incremental_model_code_payload_bytes=first.uploaded,
                                image_id=profile['local_image_id'],preexisting_original_assets=True))
    def foreign():
        names=rem("docker ps --format '{{.Names}}'").decode().splitlines()
        return sorted(n for n in names if n.startswith(('th13-cadence-','toho-th10-fa-','th11-original-')) and 'vision26-capacity' not in n)
    def main_clear():
        n=s/'native13'
        # Complete the initial Accelerator reference actors before using their gap.
        refs=all((n/f'held-five-ppo2-{seed}/receipt.json').exists() for seed in (1601363,1601364))
        held=sum((n/f'held-{arm}-{seed}/receipt.json').exists() for arm in ('five-ppo2','five-ppo3') for seed in range(1601363,1601367))
        near_fit=len(list((s/'native17').glob('train-*/receipt.json')))>=26 or (s/'policy').is_dir()
        return not foreign() and refs and (not near_fit or held==8)
    for i,wave in enumerate(waves,1):
        outfile=study/f'wave-{i:02}.json'
        if outfile.is_file():continue
        level=wave['concurrency'];seeds=wave['seeds']
        parked=time.perf_counter()
        while not main_clear():time.sleep(6)
        parked=time.perf_counter()-parked
        def collect(seed):
            pool=MeteredPool(options(level),monitor);pool.model=runtime+'/bridge/'+sha+'.ts'
            tag=f'engineering-capacity{level}-{seed}';target=local/tag
            if (target/'receipt.json').is_file():raise RuntimeError('Incomplete timed wave from a prior run: '+tag)
            start=time.perf_counter();pool.run_credit(seed)
            r=json.loads((target/'receipt.json').read_bytes());z=np.fromfile(target/'steps.bin','<f4').reshape(-1,143)
            assert r['host']=='accelerator' and r['role']=='engineering' and r['policy_sha256']==sha and r['terminal'] and r['continues']==0
            assert np.all(z[:,141]==0) and tuple(z[0,[125,127,129]])==(1,2,2) and np.all(z[:,122]==z[:,123])
            ticks=z[:,135]-z[:,134];valid=(z[:,125]==z[:,126])&(ticks>=0)&(ticks<=60)
            secs=float(z[-1,121])
            result=dict(**pool.cost,seed=seed,steps=r['steps'],native_key_matches=r['native_key_match'],
                movement_labels=np.unique(z[:,110]).astype(int).tolist(),valid_native_ticks=int(ticks[valid].sum()),tick_intervals=int(valid.sum()),
                native_tick_median=float(np.median(ticks[valid])),credit_wall_s=secs,hold_ms_percentiles=np.percentile(z[:,140]*1000,[50,95,99]).tolist(),
                actual_ingress_bytes=pool.downloaded,actual_egress_bytes=pool.uploaded,
                local_stored_bytes=sum(f.stat().st_size for f in target.iterdir() if f.is_file()),
                collector_wall_s=time.perf_counter()-start,receipt_sha256=digest(target/'receipt.json'),terminal_stage2=r['stage2'])
            save(target/'acquisition-cost.json',result);return result
        snap=lambda:json.loads(rem('python3 '+monitor+' snapshot '+runtime))
        before=snap()
        if before['free_bytes']<5*1024**3:raise OSError('Insufficient private capture capacity on Accelerator.')
        started=time.perf_counter();observed=set()
        with ThreadPoolExecutor(level) as lanes:
            futures=[lanes.submit(collect,seed) for seed in seeds]
            while not all(f.done() for f in futures):
                observed.update(foreign());time.sleep(1)
            rows=[f.result() for f in futures]
        wall=time.perf_counter()-started;after=snap()
        measured=(after['monotonic_ns']-before['monotonic_ns'])/1e9
        save(outfile,dict(**wave,engineering_only=True,measured_wave_wall_s=wall,park_before_s=parked,
                 host_before=before,host_after=after,foreign_originals_during_wave=sorted(observed),
                 mean_host_busy_cores=(after['cpu_busy_jiffies']-before['cpu_busy_jiffies'])/before['hz']/measured,
                 receipts=rows))
        print('SEALED wave',i,'concurrency',level,'whole credits',len(rows),'wall',round(wall,2),flush=True)
    summarize(study)


def summarize(study):
    groups={}
    for p in sorted(study.glob('wave-*.json')):
        wave=json.loads(p.read_bytes());groups.setdefault(wave['concurrency'],[]).append(wave)
    answer={}
    for level,waves in groups.items():
        rows=[r for w in waves for r in w['receipts']];seconds=sum(w['measured_wave_wall_s'] for w in waves)
        answer[str(level)]=dict(credits=len(rows),valid_decisions=sum(r['steps'] for r in rows),
            usable_native_key_matches=sum(r['native_key_matches'] for r in rows),wall_s=seconds,
            policy_decisions_per_s=sum(r['steps'] for r in rows)/seconds,native_acknowledged_decisions_per_s=sum(r['native_key_matches'] for r in rows)/seconds,
            native_ticks_per_s=sum(r['valid_native_ticks'] for r in rows)/seconds,credits_per_hour=3600*len(rows)/seconds,
            median_boot_to_confirmed_stock_s=float(np.median([r['qualified_boot_s']for r in rows])),
            median_per_credit_action_p50_ms=float(np.median([r['hold_ms_percentiles'][0]for r in rows])),
            median_per_credit_action_p95_ms=float(np.median([r['hold_ms_percentiles'][1]for r in rows])),
            cpu_host_average_cores=sum(w['mean_host_busy_cores']*w['measured_wave_wall_s']for w in waves)/seconds,
            game_cpu_s=sum(r['game_cgroup_cpu_usec'] for r in rows)/1e6,inference_cpu_s=sum(r['inference_cpu_s']for r in rows),
            stored_bytes=sum(r['local_stored_bytes']for r in rows),remote_folders_bytes=sum(r['disk_bytes']for r in rows),
            foreign_original_overlap=any(w['foreign_originals_during_wave'] for w in waves))
    if any(len([r for w in waves for r in w['receipts']])!=8 for waves in groups.values()):raise ValueError('Incomplete comparison')
    eligible=[l for l,r in answer.items() if not r['foreign_original_overlap'] and r['usable_native_key_matches']/r['valid_decisions']>=.98]
    best=max(eligible,key=lambda l:answer[l]['native_acknowledged_decisions_per_s']) if eligible else None
    command=(['python','-m','scripts.run_th13_accelerator_camera','--qualification','<data>/engineering/th13-accelerator',
              '--connection-config','<private-worker-config.json>','--checkpoint','<frozen-policy.ts>','--output','<new-study>/native13',
              '--difficulty','0','--role','train','--arm','<registered-arm>','--workers',best,'--seeds','<new-registered-seeds>'] if best else None)
    save(study/'throughput.json',dict(levels=answer,best_concurrency=int(best) if best else None,training_command_argv=command,
         selection_metric='Maximum end-to-end native-input-matched decisions per wall second with >=98% matching and no overlapping other original-game study.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('workers',type=Path)
    a=p.parse_args();run(a.root.resolve(),a.workers.resolve())
