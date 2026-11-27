from pathlib import Path
from scripts.th11_trial_lane import Lane,SRC
import time
l=Lane(Path("../../outputs/touhou-vision-261126"),Path("../zero-shot-pixel-native.private.json"),"calibration-hard")
l.start()
l.remote("cat >"+l.dir+"/read-calibration.c",(SRC/"read-calibration.c").read_bytes())
l.inside("""i686-w64-mingw32-gcc -O2 /capture/read-calibration.c -o /capture/calibration.exe
for n in $(seq 1 45); do xwininfo -root -tree | grep -q 208x154 && break; sleep 1; done
xdotool mousemove 427 364 click 1; sleep .12; xdotool mousemove 507 449 click 1; sleep 12
wine /capture/calibration.exe > /capture/calibration.log 2>&1 &
xdotool keydown z; sleep .15; xdotool keyup z; sleep 1
xdotool keydown Down; sleep .15; xdotool keyup Down; sleep 1
scrot -o /capture/selected-hard.png
for i in 1 2 3; do xdotool keydown z; sleep .15; xdotool keyup z; sleep 1.5; done
sleep 2
scrot -o /capture/opening-hard.png
id=10
for k in none Left Up Right Down Shift_L z x none Left Right; do
  if [ "$k" != none ]; then xdotool keydown "$k"; fi
  sleep .18; echo "$id" >/capture/calibration-request.txt
  for c in $(seq 1 25); do test -s /capture/calibration-$(printf '%03d' "$id").bin && break; sleep .02; done
  if [ "$k" != none ]; then xdotool keyup "$k"; fi
  sleep .14; id=$((id+1))
done
""",detach=True)
dest=Path("../th11-trial-calibration-private/calibration-hard");dest.mkdir(exist_ok=True,parents=True)
for i in range(55):
 if l.remote("if test -s "+l.dir+"/calibration-020.bin; then echo yes; fi").strip():break
 time.sleep(1)
for n in range(10,21):
 (dest/f"{n:03d}.bin").write_bytes(l.remote("cat "+l.dir+f"/calibration-{n:03d}.bin"))
for x in ("opening-hard.png","selected-hard.png"):
 (l.out/x).write_bytes(l.remote("cat "+l.dir+"/"+x))
print("captured",flush=True)
