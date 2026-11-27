from pathlib import Path
from scripts.th11_trial_lane import Lane,SRC
import sys
r=Path("../../outputs/touhou-vision-261126")
q=Lane(r,Path("../zero-shot-pixel-native.private.json"),sys.argv[1])
if sys.argv[2]=="observe":
 q.remote("cat >"+q.dir+"/read-calibration.c",(SRC/"read-calibration.c").read_bytes())
 print(q.inside("i686-w64-mingw32-gcc -O2 /capture/read-calibration.c -o /capture/calibration.exe").decode())
 q.inside("wine /capture/calibration.exe >/capture/calibration.log 2>&1",detach=True)
else:
 n=int(sys.argv[2])
 q.inside("echo "+str(n)+" >/capture/calibration-request.txt; sleep .08")
 print(q.remote("cat "+q.dir+"/calibration-data.txt").decode())
 b=q.remote("cat "+q.dir+f"/calibration-{n:03d}.bin")
 dest=Path("../th11-trial-calibration-private")/sys.argv[1];dest.mkdir(parents=True,exist_ok=True)
 (dest/f"{n:03d}.bin").write_bytes(b)
 q.snap("calibration-"+str(n))
