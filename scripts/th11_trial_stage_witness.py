"""Play back a publisher-trial-compatible human Hard replay to validate TH11 native stage transitions."""
import argparse
from pathlib import Path
import time
from scripts.th11_trial_lane import Lane,SRC
from scripts.run_th10_camera_cohort import digest
from scripts.collect_th10_full_action import save

def start(root,config,source):
    lane=Lane(root,config,"positive-i-hard-s1")
    assert digest(source)=="45da52b9ea37716c658f77236bae530274e7feea9790f8d07f79a584e67c7d64"
    lane.start()
    lane.remote("docker update --cpu-shares 32 --cpus 1.25 "+lane.name)
    lane.remote("cat >"+lane.dir+"/input.rpy",source.read_bytes())
    native=(SRC/"native-stock.c").read_text(encoding="utf8")
    witness=native.replace("native.bin","witness.bin").replace("record[12]","record[16]").replace(
        "record[11]=key;","record[11]=key;record[12]=globals[19];record[13]=globals[21];record[14]=globals[22];record[15]=globals[15];")
    lane.remote("cat >"+lane.dir+"/witness.c",witness.encode())
    lane.inside("""cd /capture
i686-w64-mingw32-gcc -O2 witness.c -o witness.exe
for i in $(seq 1 60); do xwininfo -root -tree | grep -q 208x154 && break; sleep .4; done
mkdir -p /tmp/th11-private/replay
cp /capture/input.rpy /tmp/th11-private/replay/th11_01.rpy
for profile in /tmp/th11-wine/drive_c/users/*; do
 test -d "$profile/AppData/Roaming" || continue
 for name in th11tr th11; do mkdir -p "$profile/AppData/Roaming/ShanghaiAlice/$name/replay"; cp input.rpy "$profile/AppData/Roaming/ShanghaiAlice/$name/replay/th11_01.rpy"; done
done
xdotool mousemove 427 364 click 1; sleep .12; xdotool mousemove 507 449 click 1; sleep 12
scrot -o /capture/stock-title.png
wine /capture/witness.exe >/capture/witness.log 2>&1 &
xdotool keydown Down; sleep .14; xdotool keyup Down; sleep .7; scrot -o /capture/selected-replay.png
xdotool keydown z; sleep .14; xdotool keyup z; sleep 1.3; scrot -o /capture/replay-list.png
xdotool keydown z; sleep .14; xdotool keyup z; sleep 1.3; scrot -o /capture/selected-stage1.png
xdotool keydown z; sleep .14; xdotool keyup z
sleep 2.3; scrot -o /capture/opening-s1.png
for i in $(seq -w 0 300); do
  date +%s%3N >>/capture/photo-ms.txt; scrot -o "/capture/frame-$i.jpg"; sleep 1.95
done
""",detach=True)
    save(lane.out/"registration.json",dict(role="engineering-only",replay_sha256=digest(source),pilot="I",native_header="0.02a",difficulty=2,selection="Replay slot 1, select Stage 1 and allow ordinary human playback to advance. No action or scoring cohort reuses this demonstration.",
         source_url="https://ir9.hatenablog.jp/entry/20080727/1217156919",native_fields="Twelve established validation fields; extra columns candidate second stage, candidate stage clock A and B, life pieces."))
    for label in ("selected-replay.png","replay-list.png","selected-stage1.png","opening-s1.png"):
        for i in range(70):
            if lane.remote("test ! -s "+lane.dir+"/"+label+" || echo yes").strip():break
            time.sleep(1)
        (lane.out/label).write_bytes(lane.remote("cat "+lane.dir+"/"+label))
    print("opening originals ready",flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("workers",type=Path);p.add_argument("source",type=Path)
    a=p.parse_args();start(a.root.resolve(),a.workers.resolve(),a.source)
