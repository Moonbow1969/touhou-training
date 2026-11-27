"""Private unmodified TH13 trial replay lane, with no production four-head actor attached."""
import hashlib
import io
from pathlib import Path
import shlex
import subprocess
import time
import numpy as np
from PIL import Image


class ReplayLane:
    def __init__(self,ident,stage,host,asset_dir,image):
        if not ident.replace("-","").isalnum() or stage not in (1,2,3):raise ValueError("Invalid replay lane")
        self.stage=stage;self.host=host;self.assets=asset_dir;self.image=image
        self.remote="/tmp/th13-trance-independent/"+ident;self.name="th13-trance-replay-"+ident

    def command(self,script,data=None,timeout=70):
        return subprocess.run(["ssh","-T","-o","BatchMode=yes","-o","StrictHostKeyChecking=yes",self.host,script],
                              input=data,check=True,capture_output=True,timeout=timeout).stdout

    def boot(self,rpy,sha):
        payload=rpy.read_bytes();assert hashlib.sha256(payload).hexdigest()==sha
        self.command("umask 077; mkdir -p "+shlex.quote(self.remote)+"; cat >"+shlex.quote(self.remote+"/input.rpy"),payload)
        launcher=Path(__file__).resolve().parents[1]/"tools/rollout-worker-probe/th13-full/trance-replay-start.sh"
        self.command("cat >"+shlex.quote(self.remote+"/run-th13.sh"),launcher.read_bytes().replace(b"\r\n",b"\n"))
        required="09818929cceea76ff4c65630e649bbccdb85d92b2e28bb4dfe6526749e6f4fb6"
        assert self.command("sha256sum "+shlex.quote(self.assets+"/th13.exe")).decode().split()[0]==required
        self.command(shlex.join(["docker","run","-d","--name",self.name,"--label","study=th13-trance-witness","--cpus","1.25","--cpu-shares","32",
             "--memory","1250m","--pids-limit","192","--network","none","-v",self.assets+":/input:ro","-v",self.remote+":/capture",
             "--entrypoint","/bin/sh",self.image,"/capture/run-th13.sh"]))

    def run(self,script,detach=False,timeout=70):
        return self.command(shlex.join(["docker","exec"]+(["-d"] if detach else [])+["-e","DISPLAY=:66",self.name,"sh","-ec",script]),timeout=timeout)

    def get(self,file):
        return self.command("cat "+shlex.quote(self.remote+"/"+file))

    def tap(self,key):
        self.run("xdotool keydown "+key+"; sleep .13; xdotool keyup "+key+"; sleep .35")

    def select(self):
        self.run("""for i in $(seq 1 76); do win=$(xdotool search --onlyvisible --name 'Ten Desires. ver 0.01a' 2>/dev/null | tail -1); if test -n "$win";then xdotool windowfocus --sync "$win";exit 0;fi; if xwininfo -root -tree | grep -q '208x154+408+316';then xdotool mousemove 427 364 click 1; sleep .15; xdotool mousemove 507 449 click 1;fi; sleep .5;done;exit 3""")
        time.sleep(12)
        for i in range(12):
            self.run("scrot -o /capture/menu.png")
            with Image.open(io.BytesIO(self.get("menu.png"))) as im:
                if np.sum(np.asarray(im.convert("RGB"))[372:389,75:132].min(-1)>185)>=18:break
            self.tap("Down")
        else:raise ValueError("Original replay title menu was not identified")
        self.tap("z");time.sleep(.6);self.tap("z");time.sleep(.65)
        for _ in range(self.stage-1):self.tap("Down")

    def stop(self):
        label=self.command("docker inspect --format '{{index .Config.Labels \"study\"}}' "+shlex.quote(self.name)).decode().strip()
        if label!="th13-trance-witness":raise ValueError("Unexpected replay lane identity")
        self.command(shlex.join(["docker","stop","-t","1",self.name])+" && "+shlex.join(["docker","rm",self.name]))
