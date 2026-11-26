"""Verify pinned state-only weights or record one full, zero-Continue stock credit."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import time

import torch

from scripts.checkpoint_storage import digest
from scripts.relative_phase_student import RelativeActor

REPO=Path(__file__).resolve().parents[1]
WORK=REPO.parent
WEIGHTS=WORK.parent/"outputs/th14-relative-phase-student-261125/student.pt"
GAME=Path(os.environ.get("ProgramFiles(x86)","C:/Program Files (x86)"))/"Steam/steamapps/common/th14/th14.exe"
REFERENCES=[WORK/"th14-menu-default-runtime-261125.png",WORK/"th14-menu-retry-runtime-261125.png"]
PINNED={WEIGHTS:"d5bfabb005ee68f2bcc2410e6649f9df528262d28c3f215a35b2c38fbc3acdde",
        REPO/"scripts/relative_phase_student.py":"f65ed3bd91e45abf0bfa1e1f05f60cb01c5ffc5562c73a9c72dfc79fe3a3c586",
        REPO/"scripts/state_route_student.py":"705bf0b51b09684d1b97d32519d80df31453b849394d9aa64dcb5a88e09f972f",
        GAME:"4609e701b677b8317e43be92041f0c7075a292182848a7c11215c3a226034f90",
        REFERENCES[0]:"1684aeb1fea7f3c39d181c6dcd1a9adfd7b063f6b892aac901a75a4cac2442c2",
        REFERENCES[1]:"7c2ca1a1ec29d46bb1088ec86e320d06a52ef05f619b25e920ff39c6709afbfe"}


def verify(seed,out):
    checked={}
    for path,expected in PINNED.items():
        actual=digest(path)
        if actual!=expected:
            raise ValueError(f"Fingerprint mismatch for {path.name}.")
        checked[path.name]=actual
    model=RelativeActor().eval()
    model.load_state_dict(torch.load(WEIGHTS,weights_only=True,map_location="cpu")["state_dict"])
    with torch.inference_mode():
        heads=model(torch.zeros(1,42),torch.zeros(1,5),torch.zeros(1),torch.tensor([[1.,0.,3.]]))
    if any(not torch.isfinite(h).all() for h in heads) or [h.shape[1] for h in heads[:3]]!=[18,2,2]:
        raise ValueError("The frozen weights failed the CPU forward check.")
    encoders=sorted((WORK/"uv-cache/archive-v0").glob("*/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-*.exe"))
    if not encoders:raise FileNotFoundError("The stock video recorder needs bundled FFmpeg.")
    cmd=[sys.executable,"-u","-m","scripts.evaluate_relative_phase_student","--weights",str(WEIGHTS),
        "--target","ending","--seeds",str(seed),"--out",str(out),"--focus-wait","90",
        "--game-over-default-reference",str(REFERENCES[0]),"--game-over-retry-reference",str(REFERENCES[1]),
        "--menu-crop","39","256","48","20","--menu-crop","39","356","118","20","--menu-crop","15","207","70","24"]
    return cmd,encoders[0],dict(seed=seed,target="ending",allows_continue=False,credits=1,source_sha256=checked,
        verification="Pinned files and offline CPU forward only; no game process or window was accessed.",
        video="TH14 game window, 30 fps, H.264, no audio",dense_recording="per-credit stock StoryArchive")


def run(args):
    out=args.out.resolve();cmd,ffmpeg,info=verify(args.seed,out)
    if args.verify_only:
        print(json.dumps(info));return
    if out.exists() and not args.resume:raise FileExistsError(out)
    if args.resume:
        report=json.loads((out/"report.json").read_bytes())
        if report["status"]=="complete":raise ValueError("This single-credit result is already complete.")
        cmd.append("--resume")
    if shutil.disk_usage(out.parent).free<4*1024**3:raise OSError("Insufficient free recording space.")
    import psutil
    import win32gui
    import win32process
    games=[p for p in psutil.process_iter(["name","exe"]) if p.info["name"] and p.info["name"].lower()=="th14.exe"
           and p.info["exe"] and Path(p.info["exe"]).resolve()==GAME.resolve()]
    if args.game_pid is not None:games=[p for p in games if p.pid==args.game_pid]
    if len(games)!=1:raise RuntimeError("Exactly one verified stock game process must match.")
    game=games[0];window=[]
    def select(hwnd,_):
        if (win32gui.IsWindowVisible(hwnd) and win32process.GetWindowThreadProcessId(hwnd)[1]==game.pid
                and "Double Dealing Character. ver 1.00b" in win32gui.GetWindowText(hwnd)):window.append(hwnd)
    win32gui.EnumWindows(select,None)
    if len(window)!=1:raise RuntimeError("The verified game must own exactly one visible stock window.")
    video=out.with_name(out.name+"-recording");video.mkdir(parents=True,exist_ok=True)
    num=1
    while any(video.glob(f"credit-{num}.*")):num+=1
    mkv=video/f"credit-{num}.mkv";mp4=mkv.with_suffix(".mp4")
    timestamp=time();exit_code=None
    with (video/f"credit-{num}.log").open("x") as log:
        capture=subprocess.Popen([str(ffmpeg),"-hide_banner","-loglevel","warning","-f","gdigrab","-framerate","30",
            "-draw_mouse","0","-i",f"hwnd={window[0]}","-an","-c:v","libx264","-preset","veryfast","-crf","23",
            "-maxrate","2500k","-bufsize","5M","-threads","2","-pix_fmt","yuv420p","-fs","1500000000","-n",str(mkv)],
            stdin=subprocess.PIPE,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            result=subprocess.run(cmd,cwd=REPO,creationflags=subprocess.CREATE_NO_WINDOW)
            exit_code=result.returncode
        finally:
            if capture.poll() is None:
                capture.stdin.write(b"q\n");capture.stdin.flush();capture.wait(timeout=30)
    if mkv.is_file() and mkv.stat().st_size>1024:
        subprocess.run([str(ffmpeg),"-loglevel","error","-i",str(mkv),"-c","copy","-movflags","+faststart","-n",str(mp4)],
            check=True,creationflags=subprocess.CREATE_NO_WINDOW)
        if mp4.stat().st_size>1024:mkv.unlink()
    info.update(start_unix=timestamp,end_unix=time(),game_pid=game.pid,evaluator_exit=exit_code,
        recorder_exit=capture.returncode,mp4=mp4.name if mp4.exists() else None)
    (video/f"credit-{num}.json").write_text(json.dumps(info,indent=2))
    if exit_code:return exit_code
    if not mp4.exists() or capture.returncode:raise RuntimeError("The scorer finished but the recording failed.")
    report=json.loads((out/"report.json").read_bytes())
    if report["status"]!="complete" or len(report["completed"])!=1:raise RuntimeError("No completed fresh credit.")
    return 0


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,required=True);parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--verify-only",action="store_true");parser.add_argument("--resume",action="store_true")
    parser.add_argument("--game-pid",type=int)
    args=parser.parse_args()
    if not 0<=args.seed<2**63:parser.error("Use a valid native policy seed.")
    sys.exit(run(args))
