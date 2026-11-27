"""Add reviewed TH16 Hard and Lunatic original replays to the five-title camera SFT."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze


def read(p):return json.loads(p.read_bytes())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','reference','public'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();d=args.data.resolve();ref=args.reference.resolve()
    root=d/'six-title-human-camera-th16';root.mkdir(exist_ok=True);dest=args.public.resolve()/root.name;dest.mkdir(exist_ok=True)
    old=d/'first-five-title-retail-sft';parent=old/'five-reviewed';report=read(parent/'report.json')
    base=read(d/'human-train-six-complete-credits/later-train-complete-roster-frozen.json')
    common=base['prior47_train']+base['qualified']+read(old/'added-reviewed-th11.json')['players']
    if [v['archive_sha256'] for v in common]!=[v['sha256'] for v in report['train']]:raise ValueError('Different five-title training sources')
    dev=read(d/'spatial-policy-study/protocol.json')['split']['held']
    held=[(d/v['file'],v['sha256']) for v in dev]
    for f in ('human-retail17-wave2-held/held-original-synchronized-frozen.json','th11-expansion/retail-original-camera/held-qualified-01-stages-frozen.json'):
        held.extend((d/v['path'],v['archive_sha256']) for v in read(d/f)['qualified'])
    if any(digest(p)!=h for p,h in held) or [h for p,h in held]!=[v['sha256'] for v in report['held']]:raise ValueError('Different development roster')
    source=d/'th16-visual-expansion';pre=source/'ranked-human-camera-preflight-v2.json'
    protocol=root/'protocol.json'
    freeze(protocol,dict(comparison_parent_sha256=digest(parent/'policy.ts'),five_source_report=digest(parent/'report.json'),
        registered_th16_sources=digest(pre),common_windows=284222,common_stages=len(common),architecture='One image encoder and four shared movement, focus, shot and Bomb action heads. TH16 season-release words remain in the original data but a season head is outside this core-action comparison.',
        sampling='Same 3000 updates of 64 windows, original initialization, augmentations, seed, replay and rank balancing as the 84-stage five-title policy. Require both registered TH16 Stage 1 sources on Hard and Lunatic. The six games share the fixed sample budget.',
        evaluation='Repeat the 19 known development stages and the three previously tested TH11 players. No reserved TH16 test player enters this fit. TH16 training metrics describe exposure only. Measure expert imitation on new original TH16 players separately.'))
    shutil.copy2(protocol,dest/protocol.name)
    manifest=source/'train-original-synchronized-frozen.json'
    print('Registered sixth work TH16; waiting for both original human cameras',flush=True)
    while not manifest.is_file():time.sleep(15)
    roster=read(manifest);new=roster['qualified']
    if (len(new)!=2 or roster['excluded'] or {v.get('difficulty') or v.get('rank') for v in new}!={'Hard','Lunatic'}
        or len({v['player_group'].casefold() for v in new})!=2 or roster['split']!='train'):raise ValueError('Both registered TH16 ranks must qualify')
    rows=common+new
    files=[d/v['path'] for v in rows]
    if any(digest(p)!=v['archive_sha256'] for p,v in zip(files,rows)):raise ValueError('Changed source image archive')
    freeze(root/'added-th16.json',dict(source_sha256=digest(manifest),players=new))
    enc=d/read(d/'ranked-multigame-sft/expanded-four-title-synchronized-fit.json')['parent']/'camera-policy.pt'
    if digest(enc)!=report['encoder'] or digest(ref)!=read(old/'protocol.json')['reference']:raise ValueError('Different common initialization or baseline')
    model=root/'policy'
    if not (model/'report.json').is_file():
        subprocess.run([sys.executable,'-u','-m','scripts.train_human_camera_policy','--encoder',str(enc),'--reference',str(ref),'--out',str(model),
                        '--steps','3000','--natural-movement-frequency','--balance-titles','--balance-difficulties','--cpu-replay','--file-backed-replay',
                        '--train',*map(str,files),'--held',*[str(p) for p,h in held]],check=True)
    fit=read(model/'report.json')
    if (len(fit['train'])!=86 or fit['torchscript_sha256']!=digest(model/'policy.ts') or fit['sft']!=digest(model/'camera-policy.pt') or
        set(fit['observed_sampled_windows_by_game_rank'])!={f'{g} {r}' for g in ('TH10','TH11','TH12','TH13','TH16','TH17') for r in ('Hard','Lunatic')}):raise ValueError('Unverified six-title fit')
    (dest/'policy').mkdir(exist_ok=True)
    for f in ('policy.ts','camera-policy.pt','report.json','loss.jsonl'):shutil.copy2(model/f,dest/'policy'/f)
    shutil.copy2(root/'added-th16.json',dest/'added-th16.json')
    print('Fitted six-title SFT',len(fit['train']),sum(v['decisions'] for v in fit['train']),fit['observed_sampled_windows_by_game_rank'],fit['torchscript_sha256'],flush=True)
    th11dev=read(d/'th11-expansion/retail-clear-boundary-review/external-held-original-synchronized-frozen.json')['qualified']
    for cohort,tests in [('development',[p for p,h in held]),('repeated-th11',[d/v['path'] for v in th11dev])]:
        subprocess.run([sys.executable,'-m','scripts.compare_human_camera_policies','--models',str(parent),str(model),'--model-names','five','six','--held',*map(str,tests),'--out',str(root/cohort)],check=True)
        shutil.copytree(root/cohort,dest/cohort,dirs_exist_ok=True)


if __name__=='__main__':main()
