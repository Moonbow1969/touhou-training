"""Open retail TH07 Normal Stage 2 at two spare lives and three Bombs using official thprac."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time
from uuid import uuid4
import numpy as np
from PIL import Image

EXE="35467eaf8dc7fc85f024f16fb2037255f151cefda33cf4867bc9122aaa2e80ca"
DAT="5fe8d772191505bf208f966da8070ee0a9e0d0ee8567a8827ef3a362bcb0c287"
TOOL="42536217464b3b9ddf27ce9d8f13f07c0a09d510831ead24d19d0882825f7315"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    driver=Path(__file__).resolve().parent
    files=[args.game_dir/"th07.exe",args.game_dir/"th07.dat",args.thprac]
    if [digest(p) for p in files]!=[EXE,DAT,TOOL]:
        raise ValueError("The private input fingerprint does not match the registered game and tool")
    host=["wsl","-d",args.wsl,"--exec"] if args.wsl else []
    def linuxpath(p):
        return subprocess.check_output([*host,"wslpath","-u",p.resolve().as_posix()],text=True).strip() if host else str(p.resolve())
    def docker(*words,**kwargs):
        return subprocess.run([*host,"docker",*words],check=True,**kwargs)
    name="moonbow-th07-practice-"+uuid4().hex[:12]
    mounts=[]
    for p,target in zip([*files,driver,args.probe,out],["/input/th07.exe","/input/th07.dat","/tool/thprac.exe","/tools","/stageprobe","/capture"]):
        mounts.extend(["-v",linuxpath(p)+":"+target+(":rw" if target=="/capture" else ":ro")])
    glyphs=np.load(driver/"menu-glyphs.npz",allow_pickle=False)
    def shell(command):
        docker("exec",name,"sh","-ec",command,stdout=subprocess.DEVNULL)
    def keys(*symbols):
        for symbol in symbols:
            if symbol not in ("Escape","Up","Down","Right","Left","z"):
                raise ValueError(symbol)
        shell(" ".join(f"xdotool keydown {symbol}; sleep .12; xdotool keyup {symbol}; sleep .53;" for symbol in symbols))
    def grab(stem):
        shell(f"scrot -o /capture/{stem}.png")
        with Image.open(out/f"{stem}.png") as img:
            return np.array(img.convert("RGB"))
    def match(image,label,box):
        x,y,w,h=box
        mask=image[y:y+h,x:x+w,:].min(2)>210
        reference=glyphs[label]
        return (mask&reference).sum()/max(1,(mask|reference).sum())
    docker("run","-d","--name",name,"--network","none","--cpus","2","--memory","1900m","--pids-limit","300","--cap-add","SYS_PTRACE","--tmpfs","/tmp:rw,exec,size=1250m","-e","DISPLAY=:73",*mounts,"--entrypoint","sh",args.image,"/tools/start.sh",stdout=subprocess.DEVNULL)
    started=time.monotonic()
    try:
        until=time.monotonic()+60
        while not (out/"ready").exists():
            if time.monotonic()>until:raise TimeoutError("Original window or PID-specific thprac activation")
            time.sleep(.25)
        # All keys in this controller go to the private X server. Down cycles the original title menu.
        time.sleep(1.4)
        for index in range(12):
            image=grab("title-latest")
            if match(image,"practice",(435,262,157,20))>.82:break
            keys("Down")
            if index==8:keys("Escape")
        else:raise ValueError("Original Practice Start highlight unavailable")
        grab("practice-start")
        keys("z")
        grab("difficulty")
        keys("z","z","z")
        image=grab("fresh-practice-overlay")
        if match(image,"shot_title",(258,54,138,13))>.75:
            keys("z")
            image=grab("fresh-practice-overlay")
        if min(match(image,"stage1",(240,125,18,13)),match(image,"warp_none",(240,147,40,15)),match(image,"default8",(373,170,24,15)))<.84:
            raise ValueError("Expected a fresh Stage 1, None, eight-spare thprac menu")
        keys("Right","Down","Down",*(["Left"]*6),"Down",*(["Left"]*5))
        image=grab("configured-practice")
        witness={label:float(match(image,label,box)) for label,box in [
            ("stage2",(240,125,18,13)),("warp_none",(240,147,40,15)),
            ("stock2",(373,170,24,15)),("stock3",(373,193,24,15))]}
        if min(witness.values())<.84:raise ValueError(f"Configured stage/stock confirmation failed: {witness}")
        keys("z")
        shell("/stageprobe >/capture/native.csv")
        grab("original-stage")
        with (out/"native.csv").open() as src:
            samples=list(csv.DictReader(src))
        tick=[int(row["stage_tick"]) for row in samples]
        if len(samples)!=120 or tick!=sorted(set(tick)) or not all(
            row["stage"]=="2" and row["rank"]=="1" and row["shot"]=="0" and row["spares"]=="2" and row["bombs"]=="3"
            and row["misses"]=="0" and row["continues"]=="0" and row["manager_tick"]==row["stage_tick"] for row in samples):
            raise ValueError("Native original stage / stock / clock validation failed")
        shell("cd /tmp/game && sha256sum th07.exe th07.dat /tool/thprac.exe >/capture/final-hashes.txt")
        assert (out/"initial-hashes.txt").read_bytes()==(out/"final-hashes.txt").read_bytes()
        record=dict(role="engineering_practice_reset",rank="Normal",team="Reimu A",stage=2,warp="None",
                    game_sha256=EXE,data_sha256=DAT,practice_tool_sha256=TOOL,source_save_present=False,continues=0,
                    native_observed=[2,3],native_first_tick=tick[0],native_last_tick=tick[-1],native_samples=len(samples),
                    total_setup_validation_seconds=time.monotonic()-started,stage_menu_glyph_iou=witness,
                    source_sha256={f:digest(out/f) for f in ("initial-files.txt","initial-hashes.txt","final-hashes.txt","target-only.tsv",
                    "loaded.txt","practice-start.png","difficulty.png","configured-practice.png","original-stage.png","native.csv")})
        (out/"qualification.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(record),flush=True)
    finally:
        docker("stop","--timeout","2",name,stdout=subprocess.DEVNULL)
        docker("rm",name,stdout=subprocess.DEVNULL)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ("game-dir","thprac","probe","out"):
        p.add_argument("--"+flag,required=True,type=Path)
    p.add_argument("--image",required=True)
    p.add_argument("--wsl",help="Docker's WSL distro, for a Windows Python driver")
    run(p.parse_args())
