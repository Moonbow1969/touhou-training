"""Run a TorchScript full-resolution policy in private original TH13 credits on a qualified host."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import shlex
import tarfile
import time

import numpy as np

from cadence_driver import Pool, GAME_HASHES, OBSERVER, digest, dump, utc
from pack_credit import pack


class CameraPool(Pool):
    def boot(self,ident,ticks):
        name='th13-cadence-'+ident+'-a'
        try:
            return super().boot(ident,ticks)
        except Exception:
            # The launcher can create a private container before its original
            # opening stock is qualified. Release that exact failed attempt.
            if self.remote("docker inspect --format '{{.Name}}' "+shlex.quote(name)+" 2>/dev/null || true").decode().strip()=='/'+name:
                self.release(name)
            raise

    def prepare(self):
        o=self.o
        shared=Path(__file__).resolve().parents[2]/"full_action"
        src=Path(__file__).resolve().parent
        sources={"camera_actor.h":shared/"camera_actor.h","camera_service.py":shared/"camera_service.py",
                 "full-trial-actor.c":src/"full-trial-actor.c","collect-credit.sh":src/"collect-credit.sh",
                 "x11_pixels.h":src.parent/"th13-pixels/x11_pixels.h"}
        bridge=o.remote_root+"/bridge"
        self.remote("umask 077; mkdir -p "+shlex.quote(bridge))
        for name,file in sources.items():
            self.remote("cat > "+shlex.quote(bridge+"/"+name),data=file.read_bytes().replace(b"\r\n",b"\n"))
        # boot() stages these files into a private mount for each process.
        self.remote(f"set -eu; cd {shlex.quote(bridge)}; cp {shlex.quote(o.prior_bridge)}/observe.exe .; "
                    f"cp {shlex.quote(o.prior_bridge)}/sft.bin .; "
                    "gcc -O2 -Wall -std=c11 -DTOHO_CAMERA_POLICY full-trial-actor.c -o actor -l:libX11.so.6 -l:libXtst.so.6 -lm")
        check=self.remote(f"cd {shlex.quote(o.assets)} && sha256sum th13.exe th13tr.dat thbgm_tr.dat").decode().splitlines()
        if [line.split()[0] for line in check]!=GAME_HASHES:
            raise ValueError("Original trial checksums differ")
        if self.remote("sha256sum "+shlex.quote(bridge+"/observe.exe")).decode().split()[0]!=OBSERVER:
            raise ValueError("Native state observer differs")
        fingerprint=digest(o.checkpoint)
        self.model=bridge+"/"+fingerprint+".ts"
        self.remote("cat > "+shlex.quote(self.model),data=o.checkpoint.read_bytes())
        if self.remote("sha256sum "+shlex.quote(self.model)).decode().split()[0]!=fingerprint:
            raise ValueError("Remote model checksum differs")

    def run_credit(self, seed):
        o=self.o;tag=f"{o.role}-{o.arm}-{seed}"
        folder=self.output/tag;folder.mkdir(exist_ok=True)
        if (folder/"receipt.json").exists():return
        for attempt in range(3):
            ident=f"vision26-{o.arm}-{seed}-{attempt}"
            try:
                self.attempt(ident,tag,folder,seed)
                return
            except Exception as exc:
                with (folder/"technical-attempts.jsonl").open("a",encoding="utf8") as log:
                    log.write(json.dumps({"id":ident,"utc":utc(),"error":str(exc)[:400]})+"\n")
                if attempt==2:raise

    def attempt(self, ident, tag, folder, seed):
        o=self.o;name,remote_dir=self.boot(ident,4);bridge=o.remote_root+"/bridge";lib=remote_dir+"/lib"
        weights=digest(o.checkpoint)
        # The socket has a short absolute host path even when the native archive path is long.
        # AF_UNIX clients inside the private mount see /capture/lib/camera.sock.
        server=shlex.join([getattr(o,'python_executable','python3'),bridge+"/camera_service.py","--model",self.model,"--sha",weights,
                           "--directory",lib,"--socket-name","camera.sock"])
        pid=None
        try:
            command=f"set -eu; cp {shlex.quote(self.model)} {shlex.quote(lib+'/policy.ts')}; "
            command+=f"touch {shlex.quote(lib+'/record-original-frames')}; cd {shlex.quote(lib)}; "
            command+=f"OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nohup {server} >camera-service.log 2>&1 </dev/null & echo $!"
            pid=int(self.remote(command).decode().strip())
            for _ in range(80):
                if self.remote(f"test -S {shlex.quote(lib+'/camera.sock')} && echo ready || true").strip():break
                time.sleep(.35)
            else:raise RuntimeError(self.remote("tail -5 "+shlex.quote(lib+"/camera-service.log")).decode())
            run=["env","TOHO_CAMERA_SHA="+weights,"TOHO_CAMERA_SOCKET=/capture/lib/camera.sock","sh",
                 "/capture/lib/collect-credit.sh",tag,str(seed),"/capture/lib/policy.ts"]
            started=utc()
            wrapper=f"set +e; {shlex.join(run)}; result=$?; echo \"$result\" > /capture/lib/collector.exit; exit \"$result\""
            self.exec(name,wrapper,True);print("ACTIVE",tag,flush=True)
            path=remote_dir+"/"+tag
            for _ in range(450):
                state=self.remote(f"if test -s {path}/settled.png; then echo done; elif test -s {lib}/collector.exit; "
                                  f"then echo failure=$(cat {lib}/collector.exit); tail -2 {lib}/camera-service.log; "
                                  f"else tail -1 {path}/actor.log 2>/dev/null || true; fi").decode().strip()
                if state=="done":break
                if "terminal=0" in state or state.startswith("failure="):
                    raise RuntimeError("Game collector ended before original terminal: "+state)
                time.sleep(3)
            else:raise TimeoutError(tag)
            for _ in range(100):
                end=self.state(remote_dir)
                if end and end[2]==1 and end[4]==-1 and end[7]==0:break
                time.sleep(.5)
            else:raise RuntimeError("Failed to verify native terminal stock")
            self.exec(name,"scrot -o /capture/"+tag+"/ranking.png; cp /capture/telemetry.bin /capture/"+tag+"/final-stock.bin")
            files=["steps.bin","events.tsv","actor.log","initial.png","ranking.png","final-stock.bin","observations.u8"]
            blob=self.remote("cd "+shlex.quote(path)+" && tar -czf - "+shlex.join(files),timeout=150)
            with tarfile.open(fileobj=io.BytesIO(blob),mode="r:gz") as archive:
                for filename in files:(folder/filename).write_bytes(archive.extractfile(filename).read())
            rollout=folder/"rollout.npz"
            record=pack(folder,o.role,seed,o.checkpoint,rollout)
            frames=np.fromfile(folder/"observations.u8",np.uint8).reshape(-1,112,96)
            if len(frames)!=record["steps"]:raise ValueError("Unpaired camera/action sequence")
            with np.load(rollout) as z:arrays={k:z[k] for k in z.files}
            np.savez_compressed(rollout,frames=frames,**arrays)
            record.update(sha256=digest(rollout),host=getattr(o,'host_label',None) or o.host,arm=o.arm,build="official TH13 Trial 0.01a",
                          game_sha256=GAME_HASHES[0],outcome="game_over",stage2_reached=record["stage2"],
                          ranking_photo="ranking.png",native_terminal_pause_state=end[2],native_ranking_tick=end[11],
                          start=started,finish=utc())
            dump(folder/"receipt.json",record)
            print("COMPLETE",tag,record["steps"],record["stage2"],flush=True)
        finally:
            if pid is not None:
                # Verify exact socket directory in the process arguments before stopping this sidecar.
                self.remote(f"if tr '\\0' '\\n' </proc/{pid}/cmdline 2>/dev/null | grep -Fqx {shlex.quote(lib)}; then kill {pid}; fi")
            self.release(name)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("host","remote-root","prior-bridge","assets","launcher","image","arm"):
        p.add_argument("--"+key,required=True)
    p.add_argument("--role",choices=("train","held"),required=True)
    p.add_argument("--known-hosts",type=Path);p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--host-label",help="Public hardware label to record in native receipts")
    p.add_argument("--python-executable",default="python3",help="Worker-local Python containing CPU PyTorch")
    p.add_argument("--output",type=Path,required=True);p.add_argument("--seeds",nargs="+",type=int,required=True)
    p.add_argument("--workers",type=int,default=3)
    opts=p.parse_args();controller=CameraPool(opts);controller.prepare()
    with ThreadPoolExecutor(opts.workers) as pool:list(pool.map(controller.run_credit,opts.seeds))
