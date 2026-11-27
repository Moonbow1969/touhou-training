from pathlib import Path
from scripts.th11_trial_lane import Lane
import sys
r=Path("../../outputs/touhou-vision-261126")
q=Lane(r,Path("../zero-shot-pixel-native.private.json"),"calibration-normal")
cmd=sys.argv[1]
if cmd=="window":
 print(q.inside("xdotool mousemove 427 364 click 1; sleep .12; xdotool mousemove 507 449 click 1; sleep 7; xwininfo -root -tree | head -12").decode())
 q.snap("official-title")
elif cmd=="menu":
 for label in ("difficulty","character","partner","opening"):
  q.tap("z");q.inside("sleep 2");q.snap(label)
elif cmd=="snap":
 q.snap(sys.argv[2])
elif cmd=="key":
 q.tap(sys.argv[2])
