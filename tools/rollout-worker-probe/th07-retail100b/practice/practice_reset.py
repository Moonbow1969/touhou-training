"""Open retail TH07 Normal Stage 2 at two spare lives and three Bombs using official thprac."""
import argparse
import csv
import hashlib
import json
import os
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
    if args.actor:
        mounts.extend(["-v",linuxpath(args.actor)+":/practice-actor:ro"])
    glyphs=np.load(driver/"scenario-glyphs.npz",allow_pickle=False)
    stage,spares,bombs,warp={"stage2-opening":(2,2,3,"None"),"letty-2-3":(1,2,3,"End Boss"),
                            "letty-0-1":(1,0,1,"End Boss"),"midboss-2-3":(1,2,3,"Mid Boss")}[args.scenario]
    worker=None
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
        if label.startswith("stock"):
            def tight(m):
                coords=np.argwhere(m)
                if not len(coords):return m[:0,:0]
                low,high=coords.min(0),coords.max(0)+1
                return m[low[0]:high[0],low[1]:high[1]]
            mask,reference=tight(mask),tight(reference)
            if mask.shape!=reference.shape:return 0.
        return (mask&reference).sum()/max(1,(mask|reference).sum())
    docker("run","-d","--name",name,"--network","none","--cpus","2","--memory","1900m","--pids-limit","300","--cap-add","SYS_PTRACE","--tmpfs","/tmp:rw,exec,size=1250m","-e","DISPLAY=:73",*mounts,"--entrypoint","sh",args.image,"/tools/start.sh",stdout=subprocess.DEVNULL)
    started=time.monotonic()
    try:
        until=time.monotonic()+110
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
        if warp=="None":
            keys("Right","Down","Down")
            stock_top=170
        elif warp=="End Boss":
            keys("Down","Right","Right","Right","Down","Down","Down")
            stock_top=216
        else:
            keys("Down","Right","Right","Down","Down")
            stock_top=193
        keys(*(["Left"]*(8-spares)),"Down",*(["Left"]*(8-bombs)))
        image=grab("configured-practice")
        check=[("stage"+str(stage),(240,125,18,13)),("stock"+str(spares),(373,stock_top,24,15)),
               ("stock"+str(bombs),(373,stock_top+23,24,15))]
        if warp=="None":check.append(("warp_none",(240,147,40,15)))
        if warp=="End Boss":check.extend([("boss_warp",(240,147,67,15)),("boss_nonspell",(240,170,107,15))])
        if warp=="Mid Boss" and "mid_warp" in glyphs:
            check.extend([("mid_warp",(240,147,67,15)),("mid_nonspell",(240,170,135,15))])
        elif warp=="Mid Boss" and args.worker:
            raise ValueError("A scored midboss deployment requires qualified literal menu witnesses")
        witness={label:float(match(image,label,box)) for label,box in check}
        if min(witness.values())<.84:raise ValueError(f"Configured stage/stock confirmation failed: {witness}")
        if args.worker:
            env=dict(os.environ,TH07_PRACTICE_CONTAINER=name,TH07_DISPLAY=":73",TH07_STOCK=str(spares)+","+str(bombs),
                     TH07_SCENARIO=args.scenario,TH07_EVIDENCE=str(out),TH07_WSL=args.wsl or "")
            worker=subprocess.Popen(args.worker,env=env)
            until=time.monotonic()+45
            while not(out/"worker-ready").exists():
                if worker.poll() is not None or time.monotonic()>until:raise RuntimeError("Worker did not arm at the private practice menu")
                time.sleep(.02)
        shell("xdotool keyup Left Right Up Down Shift_L z x; xdotool keydown z; sleep .08; xdotool keyup z")
        (out/"menu-released").touch()
        if worker:
            if worker.wait(timeout=900)!=0:raise RuntimeError("Original-game worker failed")
        else:
            shell("/stageprobe >/capture/native.csv")
        grab("original-stage")
        learned=(out/"finished-game.json").is_file()
        native_file="native-input.csv" if learned else "native.csv"
        if learned:
            with (out/native_file).open() as src:
                samples=list(csv.DictReader(src))
            live=[r for r in samples if int(r["flags"])&4 and not int(r["retry"])]
            first=live[0]
            tick=[int(r["tick"]) for r in live]
            if (int(first["lives"]),int(first["bombs"]),int(first["misses"]))!=(spares,bombs,0) or not all(
                int(r["stage"])==stage and int(r["rank"])==1 and int(r["shot"])==0 and int(r["continues"])==0 for r in live):
                raise ValueError("Original learned camera/rank/initial stock mismatch")
        else:
            with (out/"native.csv").open() as src:
                samples=list(csv.DictReader(src))
            tick=[int(row["stage_tick"]) for row in samples]
            if len(samples)!=120 or tick!=sorted(set(tick)) or not all(
                row["stage"]==str(stage) and row["rank"]=="1" and row["shot"]=="0" and row["spares"]==str(spares) and row["bombs"]==str(bombs)
                and row["misses"]=="0" and row["continues"]=="0" and row["manager_tick"]==row["stage_tick"] for row in samples):
                raise ValueError("Native original stage / stock / clock validation failed")
        (out/"build.sha256").write_text(f"{EXE}  th07.exe\n{DAT}  th07.dat\n",encoding="ascii")
        shell("cd /tmp/game && sha256sum th07.exe th07.dat /tool/thprac.exe >/capture/final-hashes.txt")
        assert (out/"initial-hashes.txt").read_bytes()==(out/"final-hashes.txt").read_bytes()
        record=dict(role="engineering_practice_reset",rank="Normal",team="Reimu A",stage=stage,warp=warp,scenario=args.scenario,
                    game_sha256=EXE,data_sha256=DAT,practice_tool_sha256=TOOL,source_save_present=False,continues=0,
                    native_observed=[spares,bombs],native_first_tick=tick[0],native_last_tick=tick[-1],native_samples=len(samples),
                    total_setup_validation_seconds=time.monotonic()-started,stage_menu_glyph_iou=witness,
                    source_sha256={f:digest(out/f) for f in ("initial-files.txt","initial-hashes.txt","final-hashes.txt","target-only.tsv",
                    "loaded.txt","practice-start.png","difficulty.png","configured-practice.png","original-stage.png",native_file)})
        (out/"qualification.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(record),flush=True)
    finally:
        if worker and worker.poll() is None:
            worker.terminate()
            worker.wait(timeout=10)
        with (out/"bootstrap.log").open("wb") as stream:
            docker("logs",name,stdout=stream,stderr=stream)
        with (out/"runtime-state.json").open("wb") as stream:
            docker("inspect","--format","{{json .State}}",name,stdout=stream)
        docker("stop","--timeout","2",name,stdout=subprocess.DEVNULL)
        docker("rm",name,stdout=subprocess.DEVNULL)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ("game-dir","thprac","probe","out"):
        p.add_argument("--"+flag,required=True,type=Path)
    p.add_argument("--image",required=True)
    p.add_argument("--wsl",help="Docker's WSL distro, for a Windows Python driver")
    p.add_argument("--scenario",choices=("stage2-opening","letty-2-3","letty-0-1","midboss-2-3"),required=True)
    p.add_argument("--actor",type=Path,help="Compiled practice-only native collector")
    p.add_argument("--worker",nargs=argparse.REMAINDER,help="Host worker argv. It must arm worker-ready, await menu-released and emit native.csv.")
    run(p.parse_args())
