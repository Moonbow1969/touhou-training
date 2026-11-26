"""Run one ending-target credit with the pinned student and optional learned contact head."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import subprocess
from time import time

import torch

from scripts import evaluate_credit_story as story
from scripts.audit_student_contact import RISK,STUDENT
from scripts.checkpoint_storage import digest
from scripts.evaluate_sft_native import Arm,output_lock
from scripts.native_story_transitions import configure
from scripts.record_relative_phase_student import GAME,PINNED,REFERENCES,REPO,WORK,verify as verify_student
from scripts.relative_phase_student import StockRelativeActor
from scripts.student_contact_policy import ContactStudent

CALIBRATION=WORK.parent/"outputs/th14-relative-student-contact-261125/calibration.json"
EXTRA={CALIBRATION:"31557b021a976eead6a42e9277241095a1f3821dce48eb2ab4d3f8b571241766",
    RISK:"7700eb95c641a1c95bdf9c4a7d51d4137a17755fe70ab673e99fff8a3301d124",
    REPO/"scripts/student_contact_policy.py":"85c97d11a069120b4d44411785055a67f68b0e362dfc76c013761dba9a30206e",
    REPO/"scripts/train_anticipatory_contact.py":"28b218965242f75579afd6a85b7edb8705c98d6fbf41b6fdce201ef10cdb44d1",
    REPO/"scripts/train_local_avoidance.py":"dde2a4ac160301af19e0e82ea6f3f571444ec081090354c668f41972ce583253"}


@contextmanager
def capture_video(encoder,directory,pid,meta):
    import psutil
    import win32gui
    import win32process
    games=[p for p in psutil.process_iter(["name","exe"]) if p.info["name"] and p.info["name"].lower()=="th14.exe"
           and p.info["exe"] and Path(p.info["exe"]).resolve()==GAME.resolve() and (pid is None or p.pid==pid)]
    if len(games)!=1:raise RuntimeError("The scorer requires one verified stock game process.")
    windows=[]
    def locate(hwnd,_):
        if win32gui.IsWindowVisible(hwnd) and win32process.GetWindowThreadProcessId(hwnd)[1]==games[0].pid and "Double Dealing Character. ver 1.00b" in win32gui.GetWindowText(hwnd):
            windows.append(hwnd)
    win32gui.EnumWindows(locate,None)
    if len(windows)!=1:raise RuntimeError("Exactly one owned TH14 window must match.")
    if shutil.disk_usage(directory.parent).free<4*2**30:raise OSError("Insufficient video reserve.")
    directory.mkdir(parents=True,exist_ok=True);n=1
    while any(directory.glob(f"credit-{n}.*")):n+=1
    raw=directory/f"credit-{n}.mkv";mp4=raw.with_suffix(".mp4");started=time()
    with (directory/f"credit-{n}.log").open("x") as log:
        proc=subprocess.Popen([str(encoder),"-hide_banner","-loglevel","warning","-f","gdigrab","-framerate","30",
            "-draw_mouse","0","-i",f"hwnd={windows[0]}","-an","-c:v","libx264","-preset","veryfast","-crf","23",
            "-maxrate","2500k","-bufsize","5M","-threads","2","-pix_fmt","yuv420p","-fs","1500000000","-n",str(raw)],
            stdin=subprocess.PIPE,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW)
        try:yield
        finally:
            if proc.poll() is None:
                proc.stdin.write(b"q\n");proc.stdin.flush();proc.wait(timeout=30)
            if raw.is_file() and raw.stat().st_size>1024:
                subprocess.run([str(encoder),"-loglevel","error","-i",str(raw),"-c","copy","-movflags","+faststart","-n",str(mp4)],
                    check=True,creationflags=subprocess.CREATE_NO_WINDOW)
                if mp4.stat().st_size>1024:raw.unlink()
            (directory/f"credit-{n}.json").write_text(json.dumps(dict(meta,started=started,ended=time(),recorder_exit=proc.returncode,
                video=mp4.name if mp4.exists() else None,game_pid=games[0].pid),indent=2))
            if proc.returncode or not mp4.exists():raise RuntimeError("Window video did not finish.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm",choices=("baseline","contact"),required=True);p.add_argument("--seed",type=int,required=True)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--verify-only",action="store_true")
    p.add_argument("--resume",action="store_true");p.add_argument("--game-pid",type=int)
    args=p.parse_args()
    if not 0<=args.seed<2**63:p.error("Use a valid native policy seed.")
    args.out=args.out.resolve();torch.set_num_threads(1)
    _,encoder,base_info=verify_student(args.seed,args.out)
    for path,fingerprint in EXTRA.items():
        if digest(path)!=fingerprint:raise ValueError(f"Fingerprint mismatch for {path.name}.")
    meta=dict(kind=args.arm,student_hash=PINNED[STUDENT],risk_hash=EXTRA[RISK],calibration_hash=EXTRA[CALIBRATION],
              frozen_contact_weight=json.loads(CALIBRATION.read_bytes())["chosen_weight"])
    actor=ContactStudent(STUDENT,RISK,CALIBRATION) if args.arm=="contact" else StockRelativeActor(STUDENT)
    if args.verify_only:
        print(json.dumps(dict(base_info,policy=meta,contact_source_hashes={p.name:v for p,v in EXTRA.items()})));return
    args.seeds=[args.seed];args.target="ending";args.limit=story.LIMITS["ending"];args.crossover=[];args.timing_policy=[]
    args.restart_hook=None;args.start_frame=30;args.focus_wait=90;args.parakit=WORK/"parakit"
    args.game_over_default_reference,args.game_over_retry_reference=REFERENCES
    args.menu_crop=[[39,256,48,20],[39,356,118,20],[15,207,70,24]];args.menu_max_error=.035
    args.transition_budget_mib=128;args.transition_reserve_mib=2048
    arms={"relative_student_"+args.arm:Arm(actor,False,meta)}
    spec=story.run_specification(args,arms);spec["controller"]["kind"]="frozen state student "+args.arm
    spec["runtime"]["source_sha256"].update({p.name:v for p,v in EXTRA.items() if p.suffix==".py"})
    spec["runtime"]["source_sha256"][__name__]=digest(Path(__file__))
    archive=configure(args,spec)
    if args.resume and json.loads((args.out/"report.json").read_bytes())["status"]=="complete":raise ValueError("Credit already complete.")
    if not args.resume:args.out.mkdir(parents=True,exist_ok=False)
    menu=story.GameOverMenu(*REFERENCES,args.menu_crop,args.menu_max_error)
    with output_lock(args.out),capture_video(encoder,args.out.with_name(args.out.name+"-recording"),args.game_pid,meta):
        result=story.evaluate(args,arms,spec,args.out/"report.json",menu,None,transition_archive=archive)
    if result["status"]!="complete" or len(result["completed"])!=1:raise RuntimeError("The fresh credit did not complete.")
    print(json.dumps(result["matched_summary"]))


if __name__=="__main__":main()
