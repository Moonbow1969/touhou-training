"""Render both verified original credits from the first registered held seed of each title."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from scripts.checkpoint_storage import digest
from scripts.render_th10_matched_credit import font
from scripts.score_original_camera_run import records


def load_source(title, receipt, data, verified, evaluation):
    trace = receipt.parent / ('trajectory.npz' if title == 11 else data.get('behavior', data.get('trajectory', data.get('file', 'missing'))))
    if digest(trace) != verified['original_trace_sha256'] or digest(receipt) != verified['native_log_sha256']:
        raise ValueError('Original data changed after scoring')
    with np.load(trace, allow_pickle=False) as z:
        actions = z['action']
        images = z['frames'] if 'frames' in z else None
        eligible = z['native_action_eligible'] if 'native_action_eligible' in z else None
    if images is None and title == 10:
        with np.load(receipt.parent/'original-pixels.npz') as z:
            images = z['obs']
        if len(images) == len(actions) + 1:
            images = images[:-1]
    if images is None and title == 12:
        images = np.fromfile(receipt.parent.parent/'raw'/data['episode']/'observations.u8', np.uint8).reshape(-1,112,96)
    if eligible is not None:
        if len(eligible)!=verified.get('raw_sampled_decisions') or eligible.sum()!=verified['decisions']:
            raise ValueError('Changed playable-action mask')
        actions=actions[eligible];images=images[eligible]
    if images is None or images.shape != (len(actions),112,96) or len(actions) != verified['decisions']:
        raise ValueError('Camera and action count differ')
    with Image.open(evaluation/verified['endpoint']) as im:
        end = im.convert('RGB')
        end.thumbnail((240,220), Image.Resampling.LANCZOS)
    return images, actions, end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    difficulties = ('Easy', 'Normal', 'Hard', 'Lunatic')
    parser.add_argument('--difficulty', choices=difficulties, help='Select the first paired seed of this registered difficulty per title.')
    parser.add_argument('--titles',type=int,nargs='+',help='Original titles included in this scored cohort.')
    args = parser.parse_args()
    evaluation = args.root/'evaluation'
    plan = json.loads((args.root/'camera-protocol.json').read_bytes())
    rank = difficulties.index(args.difficulty) if args.difficulty else None
    seeds = {int(t):min(s for s in group if rank is None or plan['rank_by_seed'][t][str(s)]==rank)
             for t, group in plan['eval_seeds'].items() if not args.titles or int(t) in args.titles}
    if args.titles and set(seeds)!=set(args.titles):
        raise ValueError('Video title is absent from the registered evaluation')
    arms = (plan['reference_arm'], plan['candidate'])
    rows = [r for r in json.loads((evaluation/'credits.json').read_bytes()) if r['seed'] == seeds[r['title']] and r['arm'] in arms]
    by_source = {(r['title'],r['seed'],r['policy_sha256']):r for r in rows}
    runs = {}
    for title, receipt in records(args.root, True, True, include_held_endpoints=True):
        data = json.loads(receipt.read_bytes())
        row = by_source.get((title,data.get('seed'),data.get('policy_sha256')))
        if row:
            key = (title,row['arm'])
            if key in runs:
                raise ValueError('Duplicate source')
            runs[key] = (row, *load_source(title,receipt,data,row,evaluation))
    order = [(title,arm) for arm in arms for title in sorted(seeds)]
    if set(runs) != set(order) or not seeds:
        raise ValueError('Missing a verified paired original credit')
    columns=len(seeds)
    width=6+264*columns
    args.out.parent.mkdir(parents=True, exist_ok=True)
    normal, small, bold = font(17), font(14), font(20,True)
    directions = ('up left','up','up right','left','neutral','right','down left','down','down right')
    fps = 15
    frames = max(len(runs[key][1]) for key in order) + fps
    with imageio.get_writer(args.out, fps=fps, codec='libx264', quality=6, macro_block_size=2,
                            output_params=['-threads','2','-movflags','+faststart']) as output:
        for i in range(frames):
            panel = Image.new('RGB',(width,718),'#14212b')
            draw = ImageDraw.Draw(panel)
            label=' · '+args.difficulty if args.difficulty else ''
            draw.text((10,6), f'{columns} original Touhou games{label}   |   {arms[0]} and {arms[1]}',font=bold,fill='white')
            draw.text((10,36),f'Decision {i+1:,}   |   15 policy decisions per playback second   |   independent game clocks',font=small,fill='#c4d6dc')
            for j,key in enumerate(order):
                row,images,actions,end = runs[key]
                x,y = 10+(j%columns)*264, 63+(j//columns)*325
                draw.text((x,y),f"TH{key[0]}  {key[1]}  ·  {row['seed']}",font=normal,fill='#bedcc4')
                if i < len(images):
                    frame = Image.fromarray(images[i]).resize((220,256),Image.Resampling.NEAREST)
                    panel.paste(frame,(x,y+26))
                    action = actions[i]
                    buttons = ' '.join(k for k,v in zip(('focus','shoot','Bomb'),action[1:]) if v)
                    draw.text((x,y+285),directions[action[0]]+'   '+buttons,font=small,fill='white')
                    if row.get('input_control_coverage')=='indeterminate':
                        draw.text((x,y+304),'Sampled input; native coverage incomplete',font=small,fill='#ffcb84')
                else:
                    panel.paste(end,(x,y+47))
                    resource = f"{row['terminal_power']:.2f} Power" if key[0] in (10,11) else f"{row['terminal_bombs']} Bombs"
                    draw.text((x,y+285),f"{row['terminal'].replace('_',' ')}  ·  {row['terminal_spares']} spares  ·  {resource}",font=small,fill='white')
                    if row.get('last_playable_bombs') is not None:
                        draw.text((x,y+304),f"Last playable stock: {row['last_playable_bombs']} Bombs",font=small,fill='white')
            output.append_data(np.asarray(panel))
    reader = imageio.get_reader(args.out)
    try:
        if reader.count_frames() != frames or reader.get_data(0).shape != (718,width,3):
            raise ValueError('Encoded video verification failed')
    finally:
        reader.close()
    args.out.with_suffix('.json').write_text(json.dumps(dict(sha256=digest(args.out),fps=fps,frames=frames,
        alignment='one decision per playback frame, original games advance independently',sources=rows),indent=2)+'\n')


if __name__ == '__main__':
    main()
