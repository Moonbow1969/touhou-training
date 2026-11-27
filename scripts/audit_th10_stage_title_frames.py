"""Rank native TH10 camera credits for manual review against a genuine Stage 2 title card."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scripts.checkpoint_storage import digest

GAME = '5430fac601d645798ccf794362ec7d4cd753814f09d160247dc628b7c6943fd8'
# Three disjoint portions of the Japanese stage caption and the small stage number.
REGIONS = ((42, 14, 55, 37), (41, 45, 55, 74), (56, 72, 76, 83))


def native_grayscale(path):
    screen = np.asarray(Image.open(path).convert('RGB'), dtype=np.uint32)
    if screen.shape != (480, 640, 3):
        raise ValueError('Expected original 640 by 480 client')
    rgb = screen[16:464, 32:416]
    weighted = rgb @ np.array([77, 150, 29], dtype=np.uint32)
    return ((weighted.reshape(112,4,96,4).sum(axis=(1,3)) + 2048)//4096).astype(np.uint8)


def score_bank(frames, template):
    if frames.dtype != np.uint8 or frames.ndim != 3 or frames.shape[1:] != (112,96):
        raise ValueError('Unexpected original camera dimensions')
    parts = [template[y:y1,x:x1] for x,y,x1,y1 in REGIONS]
    scores = np.empty(len(frames))
    for i, image in enumerate(frames):
        surfaces = [cv2.matchTemplate(image[y-4:y1+4,x-3:x1+3], ref, cv2.TM_CCOEFF_NORMED)
                    for (x,y,x1,y1),ref in zip(REGIONS,parts)]
        scores[i] = float(np.max((surfaces[0]+surfaces[1]+surfaces[2])/3))
    return scores


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('archive-root','witness','out'):
        parser.add_argument('--'+key,required=True,type=Path)
    args=parser.parse_args()
    cv2.setNumThreads(1)
    reference=native_grayscale(args.witness)
    if score_bank(reference[None],reference)[0] < .999:
        raise ValueError('Template self-match failed')
    args.out.mkdir(parents=True,exist_ok=True)
    def inspect(path):
        receipt=path.parent/'report.json'
        if not receipt.is_file():return None
        record=json.loads(receipt.read_bytes())
        if record.get('game_sha256') != GAME:return None
        with np.load(path,allow_pickle=False) as data:
            frames=data['obs']
        scores=score_bank(frames,reference)
        best=int(np.argmax(scores))
        return dict(path=path.relative_to(args.archive_root).as_posix(),camera_sha256=digest(path),
                    controller_seed=record.get('seed'),policy_sha256=record.get('policy_sha256'),
                    frames=len(frames),best_frame=best,best_score=float(scores[best]),
                    max_three_frame_mean=float(np.max(np.convolve(scores,np.ones(3)/3,'valid'))) if len(frames)>=3 else None)
    files=sorted(args.archive_root.rglob('original-pixels.npz'))
    with ThreadPoolExecutor(4) as pool: rows=[r for r in pool.map(inspect,files) if r is not None]
    rows.sort(key=lambda r:r['best_score'],reverse=True)
    unique={}
    for r in rows:
        unique.setdefault(r['camera_sha256'],r)
    lookup=list(unique.values())
    (args.out/'search.json').write_text(json.dumps(dict(game_sha256=GAME,witness_sha256=digest(args.witness),
        raw_files_considered=len(files),eligible_files=len(rows),unique_archives=len(lookup),unique_camera_frames=sum(r['frames'] for r in lookup),
        method='Maximum normalized pixel correlation over three fixed caption patches with a shared translation. For candidate discovery only. A low score is not a certified negative.',
        credits=rows),indent=2)+'\n')
    cols=5;short=lookup[:19]
    sheet=Image.new('RGB',(cols*160,4*213),'#19232b');draw=ImageDraw.Draw(sheet)
    draw.text((2,188),'Stage 2 reference',fill='white');sheet.paste(Image.fromarray(reference).resize((154,180)),(2,2))
    for j,row in enumerate(short,1):
        x=(j%cols)*160;y=(j//cols)*213
        with np.load(args.archive_root/row['path']) as z:frame=z['obs'][row['best_frame']]
        sheet.paste(Image.fromarray(frame).resize((154,180)),(x+2,y+2))
        draw.text((x+2,y+183),f"{row['controller_seed']} / {row['best_frame']}",fill='white')
        draw.text((x+2,y+197),f"match {row['best_score']:.3f}",fill='white')
    sheet.save(args.out/'top-camera-matches.jpg',quality=88)
    print('EXAMINED',len(lookup), sum(r['frames'] for r in lookup),'highest',lookup[0]['best_score'] if lookup else None,flush=True)


if __name__=='__main__':
    main()
