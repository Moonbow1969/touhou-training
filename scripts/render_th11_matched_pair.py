"""Render the first registered Hard and Lunatic TH11 original-game pairs at recorded wall speed."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import imageio_ffmpeg

from scripts.collect_th10_full_action import save
from scripts.render_th11_trial_credit import run as render
from scripts.run_th10_camera_cohort import digest


def run(study, public):
    plan = json.loads((study / "camera-protocol.json").read_bytes())
    arms = [plan["reference_arm"], plan["candidate"]]
    for rank, label in [(2, "hard"), (3, "lunatic")]:
        seed = next(s for s in plan["eval_seeds"]["11"] if plan["rank_by_seed"]["11"][str(s)] == rank)
        output = study / "native11" / "video"
        outputs = []
        evidence = []
        for arm in arms:
            source = study / "native11" / f"held-{arm}-{seed}"
            receipt = json.loads((source / "receipt.json").read_bytes())
            assert receipt["arm"] == arm and receipt["difficulty"] == rank and receipt["seed"] == seed
            trace = output / f"individual-{seed}" / arm
            render(source, trace)
            outputs.append(trace / "full-credit.mp4")
            evidence.append(dict(arm=arm,host=receipt["host"],policy_sha256=receipt["policy_sha256"],
                receipt_sha256=digest(source / "receipt.json"),endpoint_sha256=digest(source / "endpoint.png")))
        assert evidence[0]["host"] == evidence[1]["host"]
        movie = output / f"first-{label}-{seed}.mp4"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-v","error","-y","-i",str(outputs[0]),"-i",str(outputs[1]),
            "-filter_complex","hstack=inputs=2:shortest=0","-an","-c:v","libx264","-crf","18","-threads","2",
            "-movflags","+faststart",str(movie)], check=True)
        metadata = movie.with_suffix(".json")
        save(metadata,dict(seed=seed,difficulty=rank,sha256=digest(movie),actual_sources=evidence,
            playback="Recorded 1x wall time. Each original 96×112 policy camera holds samples until the next capture and ends on its own photographed Game Over screen. The shorter side holds its final screen until the other credit ends."))
        destination = public / study.name / "native11/video"
        destination.mkdir(parents=True, exist_ok=True)
        for src in (movie, metadata):
            shutil.copy2(src,destination / src.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study",type=Path)
    parser.add_argument("--public",type=Path,default=Path("results/2026-11-26-original-camera-training"))
    args = parser.parse_args()
    run(args.study,args.public)
