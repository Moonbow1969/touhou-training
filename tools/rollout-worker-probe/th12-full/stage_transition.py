"""Recognize the unmodified TH12 0.02a stage banner and highlighted result row."""
from functools import lru_cache
from pathlib import Path
import csv
import hashlib

import cv2
import numpy as np


@lru_cache(maxsize=1)
def native_templates():
    with np.load(Path(__file__).with_name("native-stage-glyphs.npz")) as archive:
        return {key: archive[key] for key in archive.files}


def dice(a, b):
    return 2 * np.count_nonzero(a & b) / max(1, np.count_nonzero(a) + np.count_nonzero(b))


def read_native_rank(path):
    """Only certify a result on the original score-ranking screen's highlighted row."""
    pixels = cv2.imread(str(path))
    if pixels is None or pixels.shape != (480, 640, 3):
        raise ValueError(f"Expected a native TH12 640×480 screenshot at {path}")
    t = native_templates()
    title = dice(pixels[58:81, 150:300].min(axis=2) > 205, t["title"])
    result = dict(stage=None, row_y=None, title_dice=title, word_dice=None, digit_dice=None)
    if title < .98:
        return result
    b, g, r = pixels.astype(np.int16).transpose(2, 0, 1)
    yellow = (np.minimum(r, g) > 165) & (b < 120) & (np.abs(r-g) < 80)
    word, y = max((dice(yellow[y-6:y+7, 336:387], t["word"]), y) for y in range(90, 275))
    digit, stage = max((dice(yellow[y-6:y+7, 391:405], t[f"digit{i}"]), i) for i in (1, 2, 3))
    return dict(stage=stage if min(word, digit) >= .98 else None,
                row_y=y if word >= .98 else None, title_dice=title, word_dice=word, digit_dice=digit)


def native_banner(path):
    """Require the Japanese Stage 2 title and its separate purple subtitle stripe."""
    pixels = cv2.imread(str(path))
    if pixels is None or pixels.shape != (480, 640, 3):
        raise ValueError(f"Expected a native TH12 640×480 screenshot at {path}")
    gray = cv2.cvtColor(pixels[80:300, 85:134], cv2.COLOR_BGR2GRAY)
    score = float(cv2.matchTemplate(gray, native_templates()["banner"], cv2.TM_CCOEFF_NORMED).max())
    hsv = cv2.cvtColor(pixels[315:341, 73:225], cv2.COLOR_BGR2HSV)
    hue, sat = hsv[:, :, 0], hsv[:, :, 1]
    purple = float(((hue >= 125) & (hue <= 165) & (sat > 80)).mean())
    return dict(detected=score >= .93 and purple >= .30, title_correlation=score, purple_fraction=purple)


def actor_banner_scores(frames):
    """Return one score per actual pre-action capture, including fades and dialogue."""
    template = native_templates()["actor_banner"]
    frames = np.asarray(frames)
    if frames.dtype != np.uint8 or frames.shape[1:] != (112, 96):
        raise ValueError("Expected uint8 [T,112,96] original observations.")
    return np.array([cv2.matchTemplate(im[15:72, 12:26], template, cv2.TM_CCOEFF_NORMED).max()
                     for im in frames], dtype=np.float32)


def native_stage_telemetry(raw):
    """Authenticate a complete live recording against the independent native loss and actor clock."""
    path = raw / "stage-native.tsv"
    if not path.is_file():
        return None
    observer_hash = (raw / "stage-sensor-sha256.txt").read_text().strip()
    expected = "4763d18112c63e1639bd8f5ce32fca0b1825fe3b489722ecb362963211450bd6"
    if observer_hash != expected:
        raise ValueError(f"Unrecognized TH12 stage observer {observer_hash}")
    with path.open() as stream:
        rows = [{k:int(v) for k,v in r.items()} for r in csv.DictReader(stream, delimiter="\t")]
    with (raw / "decisions.tsv").open() as stream:
        decisions = list(csv.DictReader(stream,delimiter="\t"))
    actor_s = (int(decisions[-1]["observed_ns"])-int(decisions[0]["onset_ns"]))/1e9
    relative_s = (rows[-1]["ms"]-rows[0]["ms"])/1000 if rows else 0
    valid = [r for r in rows if r["mode"]==0 and r["stage"]==r["manager_stage"]
             and 1<=r["stage"]<=3 and r["tick"]>=0]
    result = dict(qualified=False, max_stage=None, sensor_sha256=observer_hash,
                  trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), sampled_rows=len(rows),
                  recording_rows=len(valid), sensor_seconds=relative_s, actor_seconds=actor_s,
                  stage2_observed_ms=None, stage2_first_tick=None, entry_spares=None, entry_bombs=None)
    build_path=raw/"game-exe-sha256.txt"
    if not build_path.is_file() or build_path.read_text().split()[0]!="c6d15044394a33d6116c09eac44b33fdec814541c78518e29f60d54f49bdbd3a":
        return result
    end_path=raw/"outcome.txt"
    if not end_path.is_file() or not end_path.read_text().startswith("game_over "):
        return result
    if not any((raw/name).is_file() for name in ("terminal-at-control.ppm","endpoint.png","stage-ranking.png")):
        return result
    if len(valid)<5 or not actor_s-.35<=relative_s<=actor_s+8:
        return result
    if valid[0]["stage"]!=1 or valid[0]["spares"]!=2 or any(
            b["ms"]-a["ms"]>1000 for a,b in zip(rows,rows[1:])):
        return result
    if valid[-1]["spares"]!=-1:
        return result
    sequence = [r["stage"] for r in valid]
    if any(right-left not in (0,1) for left,right in zip(sequence,sequence[1:])):
        return result
    if any(sequence.count(i)<3 for i in range(1,max(sequence)+1)):
        return result
    if any(r["stage"]>max(sequence) for r in rows if r["mode"]==0):
        return result
    result.update(qualified=True,max_stage=max(sequence))
    stage2 = [r for r in valid if r["stage"]==2]
    if stage2:
        first=stage2[0]
        result.update(stage2_observed_ms=first["ms"],stage2_first_tick=first["tick"])
        clock_path=raw/"stage-sensor-clock.txt"
        if clock_path.is_file():
            brackets=[float(t) for t in clock_path.read_text().split()]
            if len(brackets)==2 and 0<=brackets[1]-brackets[0]<=1:
                result["observer_start_bracket_s"]=brackets
                # The first policy action after the upper bound is certainly in the new stage.
                onset=(brackets[1]+first["ms"]/1000)*1e9
                positions=[i for i,r in enumerate(decisions) if int(r["onset_ns"])>=onset]
                if positions:result["first_stage2_action"]=positions[0]
        if first["tick"]<=4:
            for key,dest in (("spares","entry_spares"),("bombs","entry_bombs")):
                if first[key]>=0 and all(r[key]==first[key] for r in stage2[:2]):
                    result[dest]=first[key]
    return result


def inspect_credit(raw):
    """Audit the score and entire available action trace without inferring an unobserved rank."""
    files = [raw / name for name in ("stage-ranking.png", "terminal-at-control.ppm", "endpoint.png")]
    ranks = [(file.name, read_native_rank(file)) for file in files if file.is_file()]
    verified_stages = {result["stage"] for _, result in ranks if result["stage"] is not None}
    if len(verified_stages) > 1:
        raise ValueError(f"Conflicting native ranking stages in {raw}")
    rank = next(iter(verified_stages), None)
    telemetry = native_stage_telemetry(raw)
    max_stage = telemetry["max_stage"] if telemetry and telemetry["qualified"] else None
    if rank is not None and max_stage is not None and rank!=max_stage:
        raise ValueError(f"Original ranking and qualified native stage trace disagree in {raw}")
    camera = raw / "observations.u8"
    action_step = None
    highest_action_score = None
    action_count = 0
    color_files = sorted((raw / "evidence").glob("*.jpg"))
    if camera.is_file() and camera.stat().st_size:
        frames = np.memmap(camera, dtype="u1", mode="r").reshape(-1, 112, 96)
        scores = actor_banner_scores(frames)
        action_count = len(scores)
        highest_action_score = float(scores.max())
        hits = scores >= .94
        consecutive = np.flatnonzero(hits[:-1] & hits[1:])
        if len(consecutive):
            action_step = int(consecutive[0])
    # Where no action camera was kept, a full search of the periodic native record is necessary.
    # For action candidates independently require the original title with its separate color bar.
    color_hits = []
    if action_count == 0 or highest_action_score >= .90 or (rank is not None and rank >= 2):
        color_hits = [p.name for p in color_files if native_banner(p)["detected"]]
    banner = bool(color_hits)
    if (rank == 1 or max_stage == 1) and banner:
        raise ValueError(f"Stage 1 native ranking disagrees with Stage 2 original banner in {raw}")
    if rank is not None:
        reached = rank >= 2
    elif max_stage is not None:
        reached = max_stage >= 2
    elif banner:
        reached = True
    else:
        reached = None
    return dict(stage2_reached=reached, rank=rank, rank_evidence=dict(ranks),native_telemetry=telemetry,
                stage2_title_files=color_hits, first_confirmed_action_step=action_step if banner else None,
                first_title_step_candidate=action_step, peak_camera_correlation=highest_action_score,
                camera_actions=action_count, periodic_captures=len(color_files))
