"""Authenticate TH12 0.02a original difficulty using its live recorder and HUD."""
import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_hud(path, templates):
    with Image.open(path) as im:
        if im.size != (640, 480):
            raise ValueError(f"Unexpected native client size in {path}")
        crop = np.asarray(im.convert("RGB").crop((485, 18, 574, 42)))
    white = crop.min(axis=2) > 209
    scores = (2 * (white & templates).sum(axis=(1, 2))
              / (white.sum() + templates.sum(axis=(1, 2))))
    best = int(scores.argmax())
    return (best if scores[best] >= .88 else None), float(scores[best])


def read_menu(path):
    with Image.open(path) as image:
        if image.size != (640,480):
            raise ValueError(f"Unexpected original menu geometry in {path}")
        text=np.asarray(image.convert("RGB").crop((190,200,450,227))).max(axis=2)<55
    with np.load(Path(__file__).with_name("difficulty-menu-glyphs.npz")) as z:
        masks=np.stack([z[str(i)] for i in (1,2,3)])
    dice=2*(masks & text).sum(axis=(1,2))/(masks.sum(axis=(1,2))+text.sum())
    index=int(dice.argmax())
    return (index+1 if dice[index]>=.98 else None),float(dice[index])


def verify_difficulty(raw, expected, credit_seconds, observer_sha):
    if expected not in (2, 3):
        raise ValueError("Expected a registered Hard or Lunatic credit")
    if (raw / "game-exe-sha256.txt").read_text().split()[0] != "c6d15044394a33d6116c09eac44b33fdec814541c78518e29f60d54f49bdbd3a":
        raise ValueError("Unexpected TH12 build")
    if (raw / "difficulty-observer-sha256.txt").read_text().strip() != observer_sha:
        raise ValueError("Unexpected original difficulty observer")
    path = raw / "difficulty-native.tsv"
    with path.open(newline="") as f:
        rows = [{k:int(v) for k,v in row.items()} for row in csv.DictReader(f, delimiter="\t")]
    live = [r for r in rows if r["mode"] == 0 and r["stage"] >= 1 and r["frame"] >= 0]
    if (len(live) < 20 or live[0]["stage"] != 1 or live[0]["spares"] != 2 or
            live[-1]["spares"] != -1 or live[-1]["ms"]/1000 < credit_seconds-2.5 or
            abs(rows[-1]["ms"]/1000-credit_seconds) > 2.5 or
            max(np.diff([r["ms"] for r in live])) > 1200):
        raise ValueError("Native difficulty observer did not cover a whole original credit")
    if any((r["rank"],r["recorded_rank"],r["character"],r["shot"]) !=
           (expected,expected,0,0) for r in live):
        raise ValueError("Native game difficulty, recorded difficulty or shot disagrees")
    with np.load(Path(__file__).with_name("difficulty-hud-glyphs.npz")) as z:
        templates = np.stack([z[str(i)] for i in range(4)])
    start = raw/"difficulty-start-hud.png"
    initial_menu,initial_score=read_menu(raw/"difficulty-menu-initial.png")
    selected,selected_score=read_menu(raw/"difficulty-menu-selected.png")
    if (initial_menu,selected)!=(1,expected):
        raise ValueError("Native Rank Select labels disagree with the registered story route")
    if read_hud(start,templates)[0] != expected:
        raise ValueError("The original start HUD disagrees with the two native difficulty fields")
    pics = sorted((raw/"evidence").glob("*.jpg"))
    visual = [(p.name,*read_hud(p,templates)) for p in pics]
    agreed = [p for p,k,score in visual if k==expected]
    if len(agreed)<3 or len(agreed)<.65*len(pics) or any(k not in (expected,None) for p,k,score in visual):
        raise ValueError("Insufficient agreement of original in-game difficulty labels")
    return dict(difficulty=expected,difficulty_verified_native_samples=len(live),
                native_difficulty_trace_sha256=digest(path),difficulty_observer_sha256=observer_sha,
                difficulty_rank_proof=dict(trace=path.name,admitted_rows=len(live),sampled_rows=len(rows),
                    global_and_recorded_rank=expected,final_spares=live[-1]["spares"],
                    trace_start_ms=live[0]["ms"],trace_end_ms=live[-1]["ms"],observer_end_ms=rows[-1]["ms"],
                    game_hud_start_sha256=digest(start),menu_initial_sha256=digest(raw/"difficulty-menu-initial.png"),
                    menu_selected_sha256=digest(raw/"difficulty-menu-selected.png"),
                    menu_initial=initial_menu,menu_selected=selected,menu_initial_dice=initial_score,menu_selected_dice=selected_score,
                    hud_readable_samples=len(agreed),hud_sample_count=len(pics),
                    hud_earliest=agreed[0],hud_latest=agreed[-1]))
