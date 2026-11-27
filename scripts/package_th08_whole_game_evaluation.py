"""Publish native TH08 whole-stock numbers, independent source evidence and original colour films."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zipfile import ZIP_STORED,ZipFile

import cv2
import imageio.v2 as imageio
import numpy as np
from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze


def read(path):return json.loads(path.read_bytes())

def film(root,pair,arms,rank,output):
    meta=[];streams=[];clocks=[]
    for arm in arms:
        info=pair['rows'][arm]
        src=root/info['original_source'];qualified=root/info['qualified_source']
        original=read(src/'receipt.json');log=qualified/'timing.jsonl';video=src/'original-observations.mp4'
        if digest(video)!=original['movie_sha256'] or digest(log)!=read(qualified/'receipt.json')['timing_sha256']:raise ValueError('Changed colour footage or clock')
        frames=[json.loads(x) for x in log.read_bytes().splitlines()]
        if frames[-1]['phase']!='terminal':raise ValueError('Expected verified first stock exhaustion')
        stream=cv2.VideoCapture(str(video));streams.append(stream)
        if not stream.isOpened() or int(stream.get(cv2.CAP_PROP_FRAME_COUNT))<len(frames):raise ValueError('Original film is incomplete')
        clocks.append(frames)
        meta.append(dict(arm=arm,seed=pair['seed'],frames_through_first_continue=len(frames),video_sha256=original['movie_sha256'],
            qualified_timing_sha256=digest(log),source_receipt_sha256=info['original_receipt_sha256'],stage=info['terminal_stage'],
            decisions=info['audit']['decisions'],terminal_spares=info['audit']['terminal_spares'],last_playable_bombs=info['audit']['last_playable_bombs']))
    n=max(map(len,clocks))
    try:
        with imageio.get_writer(output,fps=8,codec='libx264',quality=6,macro_block_size=2,output_params=['-threads','2','-movflags','+faststart']) as writer:
            boards=[None,None]
            for i in range(n+8):
                pair_images=[]
                for j,(stream,events) in enumerate(zip(streams,clocks)):
                    if i<len(events):
                        ok,frame=stream.read()
                        if not ok:raise ValueError('Truncated original recording')
                        frame=cv2.resize(frame,(512,384),interpolation=cv2.INTER_AREA)
                        elapsed=(events[i]['monotonic_ns']-events[0]['monotonic_ns'])/1e9
                        board=np.zeros((436,512,3),np.uint8);board[0:384]=frame
                        cv2.putText(board,f"TH08 {rank}   {arms[j]}   {elapsed:.1f}s",(12,406),cv2.FONT_HERSHEY_SIMPLEX,.58,(240,240,240),1,cv2.LINE_AA)
                        status='First Continue' if i==len(events)-1 else 'Original game'
                        cv2.putText(board,f"Seed {pair['seed']}   {status}",(12,426),cv2.FONT_HERSHEY_SIMPLEX,.46,(186,210,188),1,cv2.LINE_AA)
                        boards[j]=board
                    pair_images.append(boards[j])
                writer.append_data(cv2.cvtColor(np.hstack(pair_images),cv2.COLOR_BGR2RGB))
    finally:
        for stream in streams:stream.release()
    freeze(output.with_suffix('.json'),dict(sha256=digest(output),fps=8,frames=n+8,sources=meta,
           alignment='One physical observation per frame. Each original game has its own monotonic time on screen. Stop each side at the first verified Continue and hold the terminal frame.'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--public',type=Path,required=True)
    p.add_argument('--study',choices=('six-game-online','six-game-retention','six-game-return-horizon'),required=True)
    a=p.parse_args();root=a.data.resolve()/a.study;first=a.study=='six-game-online'
    source=root/('first-eight/evaluation' if first else 'th08/evaluation')
    name='th08-first-ppo' if first else a.study
    pub=a.public.resolve();dest=pub/name/'evaluation';dest.mkdir(exist_ok=True,parents=True)
    while not (source/'held-results.json').is_file():time.sleep(15)
    result=read(source/'held-results.json');arms=list(result['actor_binding'])
    if len(arms)!=2 or len(result['pairs'])!=16:raise ValueError('Incomplete original TH08 experiment')
    def publish(paths):
        subprocess.run([sys.executable,'-m','scripts.publish_original_record_bundles','--public',pub,'--repo','Moonbow1969/touhou-training','--tag','camera-training-2026-11-26',*paths],check=True)
    out=dest/'th08-whole-games.json';shutil.copy2(source/'held-results.json',out)
    stocks=['# Original TH08 full-stock evaluation','',
        'Both policies played eight paired controller seeds on each difficulty with Border Team. Each pair shares a physical host and build. Original game RNG is uncontrolled. Rank and the first Continue are verified from the original game. Unknown stage outcomes remain in the denominator.','',
        '| Rank | Policy | Stage 2 | Unknown stage | Mean spare lives | Mean last playable Bombs | Mean decisions |',
        '|---|---|---:|---:|---:|---:|---:|']
    films=[]
    for rank,cell in result['paired'].items():
        for arm in arms:
            metrics=cell['metrics'];s=cell['original_stage_evidence'][arm]
            stocks.append(f"| {rank} | {arm} | {s['confirmed_stage2']}/{cell['n']} | {s['unknown']} | {metrics['terminal_spares']['means'][arm]:.2f} | {metrics['last_playable_bombs']['means'][arm]:.3f} | {metrics['decisions']['means'][arm]:.1f} |")
        pair=min((r for r in result['pairs'] if r['rank']==rank),key=lambda r:r['slot'])
        target=dest/f'th08-first-{rank.lower()}-colour.mp4'
        if not target.with_suffix('.json').exists():film(root,pair,arms,rank,target)
        films.append(target)
    publish(films)
    stocks+=['','The films show the first registered pair of each difficulty, including the full original playfield, both HUDs and the first Continue. '+ ' · '.join(f"[Full {n} recording]({read(f.with_suffix('.release.json'))['asset']})" for n,f in zip(result['paired'],films))+' · [All 32 original scores and input records](th08-whole-games.json)']
    (dest/'TH08.md').write_text('\n'.join(stocks)+'\n',encoding='utf8')
    for i in range(0,16,4):
        archive=dest.parent/f'th08-held-pairs-{i+1:02d}-{i+4:02d}.zip'
        if not archive.is_file():
            tmp=archive.with_suffix('.partial')
            with ZipFile(tmp,'w',ZIP_STORED,allowZip64=True) as z:
                for pair in result['pairs'][i:i+4]:
                    for arm,info in pair['rows'].items():
                        src=root/info['original_source'];checked=root/info['qualified_source']
                        if digest(src/'receipt.json')!=info['original_receipt_sha256']:raise ValueError('Original audit changed')
                        if digest(src/'behavior.npz')!=read(src/'receipt.json')['behavior_sha256']:raise ValueError('Changed original behavior')
                        names=('protocol.json','receipt.json','timing.jsonl','native-observers.npz','behavior.npz','chosen-rank.png','team.png','terminal.png')
                        for directory,kind in [(src,'physical')]+([(checked,'qualified')] if checked!=src else []):
                            for f in names:
                                if (directory/f).exists():z.write(directory/f,f"{arm}-{pair['seed']}/{kind}/{f}")
                        for obs in info.get('original_ranking',{}).get('observations',[])[:2]:
                            f=obs['file'];z.write(src/f,f"{arm}-{pair['seed']}/ranking/{f}")
                    z.write(source/f"pair-{pair['slot']}.json",f"pair-{pair['slot']}.json")
            with ZipFile(tmp) as z:
                if z.testzip() is not None:raise ValueError('Corrupt TH08 original evidence package')
            tmp.replace(archive)
        publish([archive])
    print('Published complete TH08 evaluation',name,flush=True)

if __name__=='__main__':
    main()
