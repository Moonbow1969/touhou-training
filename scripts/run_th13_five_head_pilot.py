"""Run a separately compiled, unassisted five-head TH13 original without modifying four-head workers."""
import argparse
import csv
import json
from pathlib import Path
import shlex
from types import SimpleNamespace
import numpy as np

from scripts.collect_th10_full_action import save
from scripts.run_th10_camera_cohort import digest
from scripts.run_th13_accelerator_camera import CameraPool,NATIVE


def actor_source():
    source=(NATIVE/"full-trial-actor.c").read_bytes().replace(b"\r\n",b"\n").decode()
    replacements={
        "static unsigned char keys[7];":"#include \"visual-trance-head.h\"\nstatic unsigned char keys[8];",
        "bits[7]={64,128,16,32,8,1,2}":"bits[8]={64,128,16,32,8,1,2,0xa00}",
        "i<7;i++)if((old^next)":"i<8;i++)if((old^next)",
        "i<7;i++)if((unsigned char)map":"i<8;i++)if((unsigned char)map",
        "syms[7]={0xff51,0xff53,0xff52,0xff54,0xffe1,0x7a,0x78}":"syms[8]={0xff51,0xff53,0xff52,0xff54,0xffe1,0x7a,0x78,0x63}",
        "i<7;i++)keys[i]":"i<8;i++)keys[i]",
        " Display*d=XOpenDisplay(NULL);":" if(!trance_open(seed))return 29;\n Display*d=XOpenDisplay(NULL);",
        "int next=mask(a);":"int next=mask(a)|trance_step(actor_features);",
        "int consumed=post[10];":'int consumed=post[10];\n  fprintf(trance_file,"%d,%d,%.10g,%.10g,%d,%d,%d,%d,%d\\n",n,trance_bit,trance_p,trance_lp,next,xheld,consumed,pre[11],post[11]);',
        "row[122]=next;row[123]=xheld;":"row[122]=next&251;row[123]=xheld&251;",
        'fclose(trace);fclose(meta);':'fclose(trace);fclose(meta);fclose(trance_file);'}
    for a,b in replacements.items():
        if source.count(a)!=1:raise ValueError(f"Unrecognized base actor anchor: {a}")
        source=source.replace(a,b)
    return source.encode()


def run(root):
    root=root.resolve();study=root/"engineering/th13-trance";out=study/"five-head-original";out.mkdir(parents=True,exist_ok=True)
    plan=study/"human-census-and-head.json";meta=json.loads(plan.read_bytes());head=study/"visual-trance-head.bin";assert digest(head)==meta["weights_sha256"]
    policy=root/"four-game-online/policy/policy.ts";seeds=[1311351,1311352,1311353]
    save(out/"protocol.json",dict(registered_seeds=seeds,scope="All three engineering credits will run; no standard four-head study consumes these data.",
         base_policy_sha256=digest(policy),fifth_policy_sha256=digest(head),five_head_source="visual-trance-head.h",input="First 84 original-camera spatial brightness and temporal-difference features only.",
         action="Pure Bernoulli sampled from human-supervised fifth logit. X11 C is emitted for the entire sampled four-tick decision.",title=13,difficulty=2,
         repeat="Fresh ordinary Hard Reimu Stage 1, 2 spares/2 Bomb, same frozen weights; separate fifth-head RNG. No Continue or rule-driven activation."))
    class Pilot(CameraPool):
        def boot(self,ident,ticks):
            lib=self.o.remote_root+"/"+ident+"/a/lib"
            self.remote("mkdir -p "+shlex.quote(lib)+"; cat >"+shlex.quote(lib+"/trance-head.bin"),data=head.read_bytes())
            name,f=super().boot(ident,ticks)
            self.remote("cat >"+shlex.quote(f+"/trance-sensor.c"),data=(NATIVE/"observe-trance-trial.c").read_bytes())
            self.exec(name,"i686-w64-mingw32-gcc -Os -Wall -Werror /capture/trance-sensor.c -o /capture/trance-sensor.exe")
            self.exec(name,"wine /capture/trance-sensor.exe >/capture/trance-sensor.log 2>&1",True)
            return name,f
    opts=SimpleNamespace(host="momentum",host_label="momentum",known_hosts=None,remote_root="/tmp/toho-v26-trance-pilot",
        prior_bridge="/tmp/toho-th13-cadence-261125/bridge",assets="/tmp/toho-th13-261125/assets/app",launcher="/tmp/toho-v26-high-camera/src/launch.sh",
        image="toho-trial:endeavor-native-v6",python_executable="python3",checkpoint=policy,output=out,role="engineering",arm="five-head",difficulty=2)
    pool=Pilot(opts);pool.prepare();bridge=opts.remote_root+"/bridge"
    for n,body in (("visual-trance-head.h",(NATIVE/"visual-trance-head.h").read_bytes()),("five.c",actor_source())):
        pool.remote("cat >"+shlex.quote(bridge+"/"+n),data=body)
    pool.remote("cd "+shlex.quote(bridge)+" && gcc -O2 -Wall -std=c11 -DTOHO_CAMERA_POLICY five.c -o five -l:libX11.so.6 -l:libXtst.so.6 -lm && cp five actor")
    weights=np.frombuffer(head.read_bytes()[8:],dtype="<f4")
    def collect(seed):
        tag=f"engineering-five-head-{seed}";ident=f"trance-five-{seed}";folder=out/tag;folder.mkdir(exist_ok=True)
        if (folder/"fifth-qualification.json").exists():return
        pool.attempt(ident,tag,folder,seed)
        remote=opts.remote_root+"/"+ident+"/a"
        for n,source in (("fifth-head.csv",remote+"/"+tag+"/fifth-head.csv"),("native-trance.csv",remote+"/trance-observations.csv")):
            (folder/n).write_bytes(pool.remote("cat "+shlex.quote(source)))
        with (folder/"fifth-head.csv").open() as f:extra=list(csv.DictReader(f))
        with np.load(folder/"rollout.npz") as z:
            t,p0=len(extra),z["features"][:,:84].astype(np.float64)@weights[1:].astype(np.float64)+float(weights[0])
            p=1/(1+np.exp(-p0));action=np.array([int(s["sampled_c"]) for s in extra]);lp=np.where(action,np.log(p),np.log1p(-p))
            err=float(np.max(np.abs(lp-np.array([float(s["sampled_logp"]) for s in extra]))))
            assert t==len(z["action"]) and err<1e-5
            np.savez_compressed(folder/"five-head-actions.npz",action=np.column_stack([z["action"],action]).astype(np.int8),
                logp=np.column_stack([z["logp"],lp]).astype(np.float32),value=z["value"],terminated=z["terminated"],
                requested=np.array([int(s["requested_full_mask"]) for s in extra]),native=np.array([int(s["native_full_mask"]) for s in extra]))
        c_samples=[r for r in extra if int(r["sampled_c"])]
        xmatches=sum(r["requested_full_mask"]==r["x11_full_mask"] for r in extra)
        assert xmatches==t
        with (folder/"native-trance.csv").open() as f:raw=list(csv.DictReader(f))
        # Distinct onsets of a native trance state preceded by the witnessed 0x800 input and a full meter.
        acts=[]
        for pre,post in zip(raw,raw[1:]):
            if (int(pre["stage"])==int(post["stage"])==1 and int(post["native_tick"])==int(pre["native_tick"])+1
                and int(pre["active_candidate"])==0 and int(post["active_candidate"])!=0):
                tick=int(post["native_tick"])
                acts.append(dict(native_tick=tick,gauge_before=int(pre["gauge_candidate"]),state_after=int(post["active_candidate"]),
                                 native_keys=int(post["native_input"]),recent_c=any(int(s["pre_tick"])-1<=tick<=int(s["post_tick"])+1 for s in c_samples)))
        result=dict(seed=seed,role="engineering",steps=t,c_decisions=len(c_samples),sum_requested_c_probability=float(p.sum()),
            c_native_bit_matches=sum((int(s["native_full_mask"])&0xa00)==0xa00 for s in c_samples),
            full_x11_acknowledgements=xmatches,logp_rescore_max_abs=err,trance_state_onsets=acts,
            observed_same_frame_c_full_gauge_onsets=sum(x["gauge_before"]==600 and (x["native_keys"]&0x800)!=0 and x["recent_c"] for x in acts),
            post_replay_npz_sha256=digest(folder/"five-head-actions.npz"),native_trace_sha256=digest(folder/"native-trance.csv"))
        save(folder/"fifth-qualification.json",result);print(result,flush=True)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as threads:list(threads.map(collect,seeds))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);run(p.parse_args().root)
