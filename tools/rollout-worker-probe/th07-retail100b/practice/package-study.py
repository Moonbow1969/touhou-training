"""Package private-server original source evidence only after all registered outcomes are sealed."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def file_or_none(p):
    return json.loads(p.read_bytes()) if p.is_file() else None
def summarize(study, intake):
    reg=json.loads((study/'registration.json').read_bytes())
    idx=json.loads((intake/'intake-index.json').read_bytes())
    if not idx['complete'] or len(idx['rows'])!=len(reg['rows']):raise ValueError('Registered native outcomes are incomplete')
    by={r['seed']:r for r in idx['rows']}
    dest=study/'source-publication'
    dest.mkdir(exist_ok=True)
    with (dest/'qa.csv').open('w',newline='',encoding='utf-8') as fh:
        wr=csv.writer(fh);wr.writerow(['seed','scene','lane','frozen_policy_sha256','qualified','outcome','decisions','native_stage1_ticks','exact_windows','consumed_actions','spares_before','bombs_before','spares_after','bombs_after','misses_at_endpoint','bombs_used_counter','end_tick','setup_and_game_seconds','game_worker_seconds','max_observer_read_ms','p999_read_bound_ms','native_camera_sha256'])
        for plan in reg['rows']:
            seed=plan['seed']
            src=study/'raw'/str(seed)
            audit=by[seed]; receipt=file_or_none(intake/audit['receipt']) or {}
            proof=file_or_none(src/'qualification.json') or {}
            end=file_or_none(src/'finished-game.json') or {}
            rows=[]
            if (src/'native-input.csv').exists():
                with (src/'native-input.csv').open() as f:rows=list(csv.DictReader(f))
            last=rows[-1] if rows else {}
            first=next((r for r in rows if int(r.get('flags',0))&4),{})
            timings=next(csv.DictReader((src/'observer-distribution.csv').open())) if (src/'observer-distribution.csv').is_file() else {}
            wr.writerow([seed,plan['scenario'],end.get('lane'),reg['policy_sha256'],audit['qualified'],end.get('outcome'),end.get('actions'),receipt.get('native_elapsed_ticks'),receipt.get('exact_native_windows'),receipt.get('action_masks_with_native_consumption'),first.get('lives'),first.get('bombs'),last.get('lives'),last.get('bombs'),last.get('misses'),last.get('bombs_used'),last.get('tick'),proof.get('total_setup_validation_seconds'),end.get('wall_seconds'),float(timings.get('max_ns',0))/1e6,float(timings.get('p999_upper_ns',0))/1e6,digest(src/'policy-trace.npz') if (src/'policy-trace.npz').is_file() else ''])
    filenames=['policy-trace.npz','archive.sealed','events.bin','applied.csv','controls.csv','native-input.csv','native-cutoff.csv','boundary-frame.npy','build.sha256','finished-game.json','qualification.json','initial-files.txt','initial-hashes.txt','final-hashes.txt','practice-start.png','difficulty.png','configured-practice.png','original-stage.png','full-opening.png','native-continue.png','practice-record-prefix.bin','record-stage1.bin','practice-result-state.csv','native-practice-result.png','practice-result-visual.json','observer-distribution.csv','cpu-allocation.txt','preforward-cpu-allocation.json','allocation.json','target-only.tsv','loaded.txt','attempt-failed.json']
    packages=[]
    for part in range(4):
        final=dest/f'th07-normal-practice-train-source-{part+1:02d}.zip'
        temp=final.with_suffix('.pending')
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=2,allowZip64=True) as archive:
            names={}
            for f in (study/'registration.json',study/'allocation-amendment.json',study/'accelerator-allocation.json',study/'gamer-return-allocation.json',study/'physical-throughput.json',dest/'qa.csv',intake/'intake-index.json'):
                archive.write(f,f.name)
            for plan in reg['rows'][part*5:part*5+5]:
                seed=plan['seed']
                base=study/'raw'/str(seed)
                for name in filenames:
                    f=base/name
                    if f.is_file():
                        arc=f'{seed}/{name}'
                        archive.write(f,arc); names[arc]=digest(f)
                r=intake/by[seed]['receipt']
                if r.is_file():
                    archive.write(r,f'{seed}/independent-intake.json')
                    names[f'{seed}/independent-intake.json']=digest(r)
            if part==0:
                setup=study/'pre-game-setup'/'accelerator-21749209'
                for name in ('attempt-failed.json','title-latest.png','initial-hashes.txt','target-only.tsv','loaded.txt','runtime-state.json','allocation.json','cpu-allocation.txt'):
                    f=setup/name
                    if f.is_file():
                        arc='pre-game-setup/accelerator-21749209/'+name
                        archive.write(f,arc);names[arc]=digest(f)
            archive.writestr('sources-sha256.json',json.dumps(names,indent=2)+'\n')
        with zipfile.ZipFile(temp) as archive:
            if archive.testzip():raise IOError('Member CRC failure')
        temp.replace(final)
        packages.append(dict(name=final.name,sha256=digest(final),bytes=final.stat().st_size))
    (dest/'archive-manifest.json').write_text(json.dumps(dict(physical_registered=len(reg['rows']),qualified=idx['qualified_sources'],archives=packages),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);p.add_argument('intake',type=Path)
    a=p.parse_args();summarize(a.study,a.intake)
