"""Create a compact-evidence clone from a qualified local TH08 native archive study."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess


def install(source,dest):
    src=source/"app"
    subprocess.run(["sha256sum","-c","tick4-deployment.sha256"],cwd=src,stdout=subprocess.DEVNULL,check=True)
    dest.mkdir(parents=True,exist_ok=False)
    (dest/"games").mkdir()
    (dest/"deps").symlink_to((source/"deps").resolve(),target_is_directory=True)
    app=dest/"app"
    subprocess.run(["cp","-a","--reflink=auto",str(src),str(app)],check=True)
    runner=app/"scripts/run_th08_trial_camera.py"
    text=runner.read_text()
    replacements={
        "from scripts.th08_movie_queue import MovieQueue":"from scripts.th08_compact_rgb import CompactRGBQueue",
        "queued_movie=MovieQueue(root/'original-observations.mp4',async_core)":"queued_movie=CompactRGBQueue(root/'compact-rgb.npz',async_core)",
        "movie_sha256=digest(root/'original-observations.mp4')":"movie_sha256=None,compact_rgb_sha256=digest(root/'compact-rgb.npz')",
        "One captured frame per decision or transition, displayed at 8 fps. Source nanosecond timestamps in timing.jsonl.":
            "One compact 56x48 RGB crop per source observation, including transitions. Source nanosecond timestamps in timing.jsonl."}
    for old,new in replacements.items():
        if text.count(old)!=1:raise ValueError(f"Expected exactly one qualified runner site: {old}")
        text=text.replace(old,new)
    runner.write_text(text)
    module="scripts/th08_compact_rgb.py"
    shutil.copyfile(Path(__file__).with_name("th08_compact_rgb.py"),app/module)
    members={r.split("  ",1)[1] for r in (src/"tick4-deployment.sha256").read_text().splitlines()}
    members.add(module)
    (app/"tick4-deployment.sha256").write_text("".join(f"{hashlib.sha256((app/name).read_bytes()).hexdigest()}  {name}\n" for name in sorted(members)))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qualified-study",required=True,type=Path);p.add_argument("--destination",required=True,type=Path)
    a=p.parse_args();install(a.qualified_study.resolve(),a.destination.resolve())
