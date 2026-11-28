"""Check archived retail human camera pixels and four-tick key alignment, without a game process."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from replays.inspect_rpy import action_streams


PLAYFIELD = (32, 16, 416, 464)
PROJECTILES = [(128, 107), (154, 116), (167, 130)]  # Centers picked on original RGB capture.


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def move(word):
    bits = np.asarray(word)
    return 4 + ((bits & 128) != 0).astype(int) - ((bits & 64) != 0).astype(int) + 3 * (
        ((bits & 32) != 0).astype(int) - ((bits & 16) != 0).astype(int))


def direction_hold_cost(source, ticks):
    # Nonoverlapping windows from the same eight-tick block starts across all three settings.
    starts=ticks[ticks%8==0]
    directions=move(source[starts[:,None]+np.arange(8)])
    result={}
    for duration in (2,4,8):
        windows=directions.reshape(-1,duration)
        counts=np.stack([(windows==direction).sum(1) for direction in range(9)],axis=1)
        result[duration]=dict(native_ticks=int(windows.size), windows=len(windows),
            hold_first_direction_disagreements=int((windows!=windows[:,:1]).sum()),
            best_constant_direction_min_disagreements=int((duration-counts.max(1)).sum()))
    return result


def timing(archive, source, receipt):
    ticks, native = archive["native_frame"].astype(int), archive["native_input"]
    bits = 0xF3 | receipt["focus_bit"]
    offsets = {}
    for lag in range(-2, 2):
        decoded = source[ticks + lag] & bits
        live = native & bits
        moving = ((decoded | live) & 0xF0) != 0
        offsets[str(lag)] = dict(matches=int((decoded[moving] == live[moving]).sum()), moving_rows=int(moving.sum()))
    np.testing.assert_array_equal(native & bits, source[ticks - 1] & bits)
    future = source[ticks[:, None] + np.arange(4)]
    np.testing.assert_array_equal(archive["input_chunk"], future)
    dirs = move(future)
    np.testing.assert_array_equal(archive["action"][:, 0], dirs[:, 0])
    same = dirs == dirs[:, :1]
    changed = ~same.all(1)
    return dict(archived_rows=len(ticks), first_last_tick=[int(ticks[0]), int(ticks[-1])],
                tick_gap_histogram=dict(sorted(Counter(map(int,np.diff(ticks))).items())),
                tick_mod_four=dict(sorted(Counter(map(int,ticks%4)).items())), moving_button_offset_test=offsets,
                contemporaneous_native_word_equals_source_at_tick_minus_one_rows=len(ticks),
                signed_control_mask=bits, future_direction_not_constant_rows=int(changed.sum()),
                future_direction_not_constant_fraction=float(changed.mean()),
                four_tick_first_direction_hold_disagrees_ticks=int((~same).sum()),
                future_source_ticks=int(same.size),
                four_tick_first_direction_hold_agreement=float(same.mean()),
                matched_eight_tick_blocks_by_hold_duration=direction_hold_cost(source,ticks))


def local_stats(rgb, camera, x, y):
    ay, ax = np.ogrid[:rgb.shape[0], :rgb.shape[1]]
    bx, by = ax-x, ay-y
    crop = rgb[y-7:y+8,x-7:x+8]
    white = (crop.min(2) >= 235) & (crop.max(2)-crop.min(2) < 25)
    iy, ix = np.where(white)
    display = np.asarray(Image.fromarray(rgb).convert("L"), dtype=float)
    surrounding = (bx**2+by**2 >= 10**2) & (bx**2+by**2 < 14**2)
    radius = 4
    # Compare original coordinate at each actor-pixel center. Camera pixels are 4x4 original pixels.
    cy, cx = np.ogrid[:112,:96]
    dx = 32 + 1.5 + 4*cx - x
    dy = 16 + 1.5 + 4*cy - y
    inside = dx**2 + dy**2 < (radius+2)**2
    outside = (dx**2+dy**2 >= 10**2) & (dx**2+dy**2 < 14**2)
    peak = int(camera[inside].max())
    return dict(original_center_xy=[x,y],actor_cell_xy=[(x-32)//4,(y-16)//4],
                original_white_core_width_height=[int(np.ptp(ix))+1,int(np.ptp(iy))+1],
                original_white_core_pixels=int(white.sum()),camera_cells_above_235_in_6px_radius=int((camera[inside]>=235).sum()),
                native_luma_peak=int(display[y-3:y+4,x-3:x+4].max()), native_background_median=float(np.median(display[surrounding])),
                camera_luma_peak=peak,camera_background_median=float(np.median(camera[outside])),
                camera_peak_above_background=peak-float(np.median(camera[outside])))


def figure(out, rgb, camera):
    region=(104,85,190,151)
    native=rgb.crop(region).resize((344,264),Image.Resampling.NEAREST)
    # Draw matched crops after coordinate mapping on a native-sized camera view.
    expanded=camera.resize((384,448),Image.Resampling.NEAREST)
    camera_crop=expanded.crop((72,69,158,135)).resize((344,264),Image.Resampling.NEAREST)
    sheet=Image.new("RGB",(730,342),"#19222a")
    draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype("arial.ttf",15) if Path("C:/Windows/Fonts/arial.ttf").is_file() else ImageFont.load_default()
    draw.text((14,8),"TH10 retail Normal, held human Stage 1, native tick 1200",font=font,fill="white")
    for col,(img,label) in enumerate(((native,"Original RGB"),(camera_crop,"Actual policy camera, nearest-neighbor display"))):
        pos=14+col*360
        draw.text((pos,34),label,font=font,fill="white");sheet.paste(img,(pos,58))
        for i,(x,y) in enumerate(PROJECTILES,1):
            xx=pos+(x-region[0])*4;yy=58+(y-region[1])*4
            draw.text((xx-5,yy-48),str(i),font=font,fill="#ffff55",stroke_width=1,stroke_fill="black")
    sheet.save(out/"same-frame-projectiles.png")


def run(held, raw, out):
    out.mkdir(parents=True,exist_ok=True)
    replay=raw/"th10-gensokyo-2749.rpy"
    streams=dict(action_streams(replay,10,version="1.00a"))
    stages={}
    pairs=[]
    for stage in (1,2):
        path=held/f"th10-gensokyo-2749-s{stage}.npz"
        receipt=json.loads(path.with_suffix(".json").read_bytes())
        assert receipt["sha256"]==digest(replay) and receipt["archive_sha256"]==digest(path)
        assert receipt["split"]=="held" and receipt["character_in_header"]=="ReimuA" and receipt["mode"]=="MoF Normal"
        with np.load(path,allow_pickle=False) as archive:
            source=np.frombuffer(streams[stage],"<u2").reshape(-1,3)[:,0]
            stages[stage]=dict(archive_sha256=digest(path),**timing(archive,source,receipt))
            if stage!=1:continue
            frames, ticks=archive["obs"],archive["native_frame"]
            for index in (32,300):
                original=raw/f"th10-slot2-stage1-{index}.png"
                rgb=Image.open(original).convert("RGB")
                assert rgb.size==(640,480)
                camera=rgb.crop(PLAYFIELD).convert("L").resize((96,112),Image.Resampling.BOX)
                rows=np.flatnonzero((frames==np.asarray(camera)).all((1,2)))
                if len(rows)!=1:raise ValueError(f"{original.name} matched {len(rows)} actor frames")
                row=int(rows[0]);tick=int(ticks[row]);dest=out/f"retail-full-original-tick-{tick}.png"
                shutil.copy2(original,dest)
                camera_file=out/f"actual-camera-tick-{tick}.png"
                camera.save(camera_file)
                matched=dict(stage=stage,source_row=row,native_tick=tick,exact_pixel_matches=len(rows),
                             source_file=dest.name,source_sha256=digest(dest),camera=camera_file.name,
                             camera_sha256=digest(camera_file),player_native_xy=archive["xy"][row].tolist())
                if tick==1200:
                    matched["illustrative_projectiles"]=[local_stats(np.asarray(rgb).astype(np.uint8),np.asarray(camera),x,y) for x,y in PROJECTILES]
                    figure(out,rgb,camera)
                pairs.append(matched)
    report=dict(game="TH10 Steam retail v1.00a",source_player="Duckator",replay_sha256=digest(replay),
                camera="Crop (32,16,416,464), convert to luminance, box average 4x4, uint8 112x96.",
                stages=stages,matched_full_originals=pairs,
                notes=["Three visually identified white-core blue projectiles in one exact matched image. They are not native object-ID labels. No collision-time full-resolution sequence is retained with this source.",
                       "Input alignment measures decoded held words against the replayed game's observed keyboard word. It does not measure an external key injector, display or human reaction latency.",
                       "Direction-hold mismatch compares the actual four recorded human tick directions with a hypothetical perfect first-direction choice held for four ticks, on captured states only. It is not a closed-loop score.",
                       "Matched-duration sensitivity uses identical eight-tick source blocks at archived ticks divisible by eight, partitioned into holds of two, four or eight ticks. Best constant direction chooses the most frequent HUMAN direction in each hold with access to its whole future, a lower bound on key mismatch on that recorded path."])
    (out/"audit.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--held",type=Path,required=True)
    p.add_argument("--raw",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.held,a.raw,a.out)
