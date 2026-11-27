"""Start the unmodified publisher-provided TH11 0.02a trial inside a private Xvfb/Wine game."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

from scripts.collect_th10_full_action import save
from scripts.run_th10_camera_cohort import digest
SRC=Path(__file__).resolve().parents[1]/"tools/rollout-worker-probe/th11-trial"


class Lane:
    def __init__(self,root,connection,ident,out=None):
        assert ident.replace("-","").isalnum()
        self.root=root.resolve();self.out=out if out is not None else self.root/"engineering/th11-trial"/ident;self.out.mkdir(parents=True,exist_ok=True)
        self.ssh=json.loads(connection.read_bytes())["momentum"]["command"][:-1]
        self.base="/tmp/toho-v26-th11-demo";self.dir=self.base+"/lanes/"+ident;self.name="th11-original-"+ident
    def remote(self,cmd,data=None,timeout=60):return subprocess.run([*self.ssh,cmd],input=data,capture_output=True,check=True,timeout=timeout).stdout
    def inside(self,cmd,detach=False,timeout=80):return self.remote(shlex.join(["docker","exec"]+(["-d"] if detach else [])+["-e","DISPLAY=:66","-e","WINEPREFIX=/tmp/th11-wine",self.name,"sh","-ec",cmd]),timeout=timeout)
    def snap(self,label):
        self.inside("scrot -o /capture/"+label+".png");b=self.remote("cat "+shlex.quote(self.dir+"/"+label+".png"));(self.out/(label+".png")).write_bytes(b);return b
    def tap(self,key,hold=.15):
        self.inside("xdotool keydown "+shlex.quote(key)+"; sleep "+str(hold)+"; xdotool keyup "+shlex.quote(key))
    def start(self):
        self.remote("mkdir -p "+shlex.quote(self.dir)+"; cat >"+shlex.quote(self.dir+"/launch-game.sh"),(SRC/"launch-game.sh").read_bytes().replace(b"\r\n",b"\n"))
        assets=self.base+"/official/app"
        sums=self.remote("cd "+assets+" && sha256sum th11.exe th11tr.dat thbgm_tr.dat").decode().splitlines()
        files={x.split()[1]:x.split()[0] for x in sums}
        manifest=dict(installer_sha256="1f41c871cec5069aa5ecc72543a331d89bfbfc69b4d64f5f3d91df5c12203853",installer_bytes=139070807,
             publisher="https://www16.big.or.jp/~zun/html/th11dl.html",listed_mirror="https://mirror.studio-ramble.com/upload/535/200806/th11tr002a_setup.exe",
             official_public_trial="0.02a; original game permits Stages 1–3",asset_sha256=files,image="toho-trial:endeavor-native-v6",host="momentum",
             private_game_assets="Local research run, original asset binaries excluded from public artifacts.")
        save(self.out/"build.json",manifest)
        self.remote(shlex.join(["docker","run","-d","--rm","--name",self.name,"--label","study=th11-original-trial","--network","none","--cpus","2",
            "--memory","1200m","--pids-limit","192","-v",assets+":/input:ro","-v",self.dir+":/capture","--entrypoint","sh",manifest["image"],"/capture/launch-game.sh"]))
        return files


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("workers",type=Path);p.add_argument("ident")
    a=p.parse_args();lane=Lane(a.root,a.workers,a.ident);print(lane.start(),flush=True);time.sleep(12)
    print(lane.inside("xwininfo -root -tree | tail -22").decode());lane.snap("startup")
