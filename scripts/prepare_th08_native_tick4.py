"""Install the native four-tick runner in a new directory beside a verified TH08 0.02 Linux study."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess


ROOT=Path(__file__).resolve().parents[1]
SOURCES=ROOT/"tools/rollout-worker-probe"
EXE="3a6b4a0de24f916e669cf642ffd4f7c22ada1d319d9ce8ca8494fe96a2f5f5c1"


def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError(f"Expected one verified runner fragment {old!r}")
    return text.replace(old,new)


def install(origin,dest,wine_image,policy,history):
    if hashlib.sha256((origin/"app/assets/game/th08tr.exe").read_bytes()).hexdigest()!=EXE:
        raise ValueError("Expected stock TH08 trial 0.02")
    if not (origin/"app/title-reference.png").is_file() or not (origin/"deps").is_dir():
        raise ValueError("Supply a prepared and qualified 0.02 study with title reference and CPU dependencies")
    dest.mkdir(parents=True,exist_ok=False)
    (dest/"deps").symlink_to((origin/"deps").resolve(),target_is_directory=True)
    (dest/"games").mkdir()
    app=dest/"app"
    subprocess.run(["cp","-a","--reflink=auto",str(origin/"app"),str(app)],check=True)
    if policy:shutil.copyfile(policy,app/"policy.ts")
    native=app/"tick4";native.mkdir()
    for name in ("reader.c","bracket.c","gate.h","stream.c"):
        shutil.copyfile(SOURCES/"th08-trial002/tick4"/name,native/name)
    shutil.copyfile(SOURCES/"th13-pixels/x11_pixels.h",native/"x11_pixels.h")
    subprocess.run(["gcc","-std=c11","-O2","-static","bracket.c","-o","bracket"],cwd=native,check=True)
    subprocess.run(["gcc","-std=c11","-O2","stream.c","-o","stream","-l:libX11.so.6","-l:libXtst.so.6"],cwd=native,check=True)
    subprocess.run(["docker","run","--rm","--pull","never","--network","none","-v",f"{native}:/clock",
        "--entrypoint","sh",wine_image,"-ec",
        "cd /clock; ! ldd ./stream | grep -q not.found; i686-w64-mingw32-gcc-posix -O2 -Wl,--no-insert-timestamp reader.c -o clock.exe"],check=True)
    entry=app/"game-entry-v2.sh"
    text=replace_once(entry.read_text(),"snap team\ntap z; sleep 1",
        "snap team\ncp /clock/clock.exe /capture/clock.exe\n/clock/bracket >/capture/clock.out 2>/capture/clock.err &\ntap z; sleep 1")
    entry.write_text(text)
    post=app/"postcredit-ranking.sh"
    post.write_text(replace_once(post.read_text(),"/input/stream","/clock/stream\nsleep .8\n"
        "if ! test -s /capture/terminal.png; then scrot -o /capture/engineering-last.png; exit 0; fi"))
    # The game process never shares its address space with the model. Preserve local FIFOs.
    runner=(ROOT/"scripts/run_th08_trial_camera.py").read_text()
    a=runner.index("    name=f'toho-th08-");b=runner.index("    logs=[]",a)
    runner=runner[:a]+'''    cmd=['sh','-c','cat /capture/frames.fifo & child=$!; cat > /capture/actions.fifo; wait "$child"']
'''+runner[b:]
    (app/"scripts/run_th08_trial_camera.py").write_text(runner)
    driver=app/"credit-v2.sh";text=driver.read_text()
    text=replace_once(text,'-v "$out:/capture" --entrypoint sh '+wine_image,'-v "$app/tick4:/clock:ro" -v "$out:/capture" --entrypoint sh '+wine_image)
    if "--history-states" in text:
        import re
        text,n=re.subn(r"--history-states [14]","",text)
        if n!=1:raise ValueError("Ambiguous original camera history length")
    text=replace_once(text,'--role "$role" --verified-menu',
        '--role "$role" --verified-menu --native-tick4 --history-states '+str(history)+' "${extra[@]}"')
    text=replace_once(text,'out="$study/games/$seed"','''extra=()
if [[ -n "${5:-}" ]]; then
  [[ "$role" == engineering-zero-shot-original && "$5" =~ ^[1-9][0-9]*$ ]]
  extra=(--engineering-decisions "$5")
fi
out="$study/games/$seed"''')
    # The Bash wrapper carries optional CLI elements without shell word splitting.
    (app/"credit-tick4.sh").write_text(text.replace("#!/bin/sh","#!/bin/bash",1))
    (app/"credit-tick4.sh").chmod(0o755)
    paths=[*sorted(native.iterdir()),entry,post,app/"scripts/run_th08_trial_camera.py",app/"credit-tick4.sh",app/"policy.ts",app/"assets/game/th08tr.exe"]
    (app/"tick4-deployment.sha256").write_text("".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(app)}\n" for p in paths))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-study",type=Path,required=True)
    p.add_argument("--destination",type=Path,required=True)
    p.add_argument("--wine-image",required=True);p.add_argument("--policy",type=Path)
    p.add_argument("--history-states",choices=(1,4),type=int,default=1)
    a=p.parse_args();install(a.source_study.resolve(),a.destination.resolve(),a.wine_image,a.policy,a.history_states)
