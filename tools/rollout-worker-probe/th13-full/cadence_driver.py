"""Run preregistered four/eight native-tick SFT credits on a qualified remote TH13 host."""
import argparse
import hashlib
import io
import json
import shlex
import struct
import subprocess
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from pack_credit import pack

GAME_HASHES=["09818929cceea76ff4c65630e649bbccdb85d92b2e28bb4dfe6526749e6f4fb6","28e1a93c85bc65def7d93338c7496cac8d2f62c07368dbb0864611e966e7927b","308465a13dd3969b55b3618385a313944f34c0f57435ea73fdf85d360133475b"]
SFT="6cee156a59339339145b441445ba86d2537620d2359be8de8075fae10e0aae58"
ACTOR="0990ebb9246bdafd28b120547255e17af7fc591455cc7b02a14e896c574a4909"
OBSERVER="6776abe37213f57811318eda02b755689cc4d9ea87faec755728146b8b77bb93"

def utc():return datetime.now(timezone.utc).isoformat()
def dump(path,d):
    temp=path.with_suffix('.partial');temp.write_text(json.dumps(d,indent=2)+'\n',encoding='utf8');temp.replace(path)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

class Pool:
    def __init__(self,opts):
        self.o=opts;self.output=opts.output;self.output.mkdir(parents=True,exist_ok=True)
        self.ssh=['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=3']
        if opts.known_hosts:self.ssh+=['-o','UserKnownHostsFile='+str(opts.known_hosts)]
        self.ssh+=[opts.host]
    def remote(self,command,data=None,timeout=45):
        result=subprocess.run(self.ssh+[command],stdin=subprocess.DEVNULL if data is None else None,input=data,capture_output=True,timeout=timeout)
        if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace')[-800:]+' (exit '+str(result.returncode)+')')
        return result.stdout
    def exec(self,name,code,detach=False):
        cmd=['docker','exec']+(['-d'] if detach else [])+['-e','DISPLAY=:66',name,'sh','-ec',code]
        return self.remote(shlex.join(cmd))
    def gate(self):
        o=self.o
        image=json.loads(self.remote('docker image inspect '+shlex.quote(o.image)))[0]
        self.image_id=image['Id']
        layers=hashlib.sha256('\n'.join(image['RootFS']['Layers']).encode()).hexdigest()
        cfg=hashlib.sha256(json.dumps(image['Config'],separators=(',',':'),sort_keys=True).encode()).hexdigest()
        if image['Os']!='linux' or image['Architecture']!='amd64' or layers!='fa091c18feede434514d430c66fd0453df321298680eec1aa64ce790e83e50f2' or cfg!='d0bada7af684aa2cab459bb260723f581338063d16d52f9f3a59719b2676d153':
            raise ValueError('Wine image layer or runtime configuration mismatch')
        hashes=self.remote(f'cd {shlex.quote(o.assets)} && sha256sum th13.exe th13tr.dat thbgm_tr.dat').decode().splitlines()
        if [row.split()[0] for row in hashes]!=GAME_HASHES:raise ValueError('Original assets differ')
        hashes=self.remote(f'cd {shlex.quote(o.remote_root+"/bridge")} && sha256sum actor observe.exe sft.bin').decode().splitlines()
        if [row.split()[0] for row in hashes]!=[ACTOR,OBSERVER,SFT] or digest(o.checkpoint)!=SFT:raise ValueError('Actor, observer or frozen policy mismatch')
        self.remote(f'cat > {shlex.quote(o.remote_root+"/bridge/collect-credit.sh")}',data=Path(__file__).with_name('collect-credit.sh').read_bytes().replace(b'\r\n',b'\n'))
    def state(self,folder):
        raw=self.remote('cat '+shlex.quote(folder+'/telemetry.bin')+' 2>/dev/null || true')
        if len(raw)!=84:return None
        s=struct.unpack('<21i',raw);return s if s[0]==s[-1] else None
    def boot(self,identifier,ticks):
        prefix='th13-cadence-'+identifier+'-';name=prefix+'a'
        base=self.o.remote_root+'/'+identifier;folder=base+'/a'
        self.remote(f'mkdir -p {shlex.quote(folder+"/lib")}; cp {self.o.remote_root}/bridge/actor {self.o.remote_root}/bridge/observe.exe {self.o.remote_root}/bridge/sft.bin {self.o.remote_root}/bridge/collect-credit.sh {shlex.quote(folder+"/lib/")}; printf "%d\\n" {int(ticks)} > {shlex.quote(folder+"/lib/decision-ticks")}')
        args=[self.o.launcher,self.o.assets,base,self.o.image,prefix]
        self.remote('TH13_LANES=a sh '+shlex.join(args),timeout=180)
        self.exec(name,'for i in $(seq 1 50); do wine /capture/lib/observe.exe >/capture/observer.log 2>&1 && break; sleep 1; done',True)
        restarts=0
        for i in range(65):
            s=self.state(folder)
            if s and s[1]==7 and s[2]==0 and s[3]==1 and s[4]==2 and s[5]==2 and s[7]==0 and s[8]==1:return name,folder
            if s and s[1]==7 and s[2]==0 and s[3]==1 and s[7]==0 and s[8]==1 and restarts<2:
                # A slow Wine startup can reach gameplay before the observer
                # attaches. Use the original pause-menu restart to recover stock.
                self.exec(name,'xdotool keydown r; sleep .18; xdotool keyup r; sleep 1.8')
                restarts+=1
            elif s and s[1]==7 and s[2]==2 and s[3]==1:self.exec(name,'xdotool keydown Escape; sleep .18; xdotool keyup Escape')
            elif s and s[1]==4 and i%2==0:self.exec(name,'xdotool keydown z; sleep .18; xdotool keyup z')
            time.sleep(1.2)
        raise RuntimeError(f'Paused Normal Stage 1 not qualified in {name}: {s}')
    def release(self,name):
        if not name.startswith('th13-cadence-') or self.remote("docker inspect --format '{{.Name}}' "+shlex.quote(name)).decode().strip()!='/'+name:raise ValueError('Unexpected capture container')
        self.remote('docker stop -t 2 '+shlex.quote(name)+' >/dev/null && docker rm '+shlex.quote(name)+' >/dev/null')
    def qualification(self):
        path=self.output/'host-qualification.json'
        if path.is_file():
            prior=json.loads(path.read_bytes())
            if prior['ssh_host']!=self.o.host or prior['actor_sha256']!=ACTOR:raise ValueError('Different host already owns the cadence cohort')
            return
        name,folder=self.boot('preflight',4)
        s0=self.state(folder);self.exec(name,'xdotool keydown r; sleep .18; xdotool keyup r; sleep 4; xdotool keydown Left z; sleep .39')
        s1=self.state(folder)
        self.exec(name,'xdotool keyup Left; xdotool keydown Right Up Shift; sleep .39')
        s2=self.state(folder)
        self.exec(name,'xdotool keyup Right Up Shift z; xdotool keydown x; sleep .36')
        s3=self.state(folder);self.exec(name,'xdotool keyup x; scrot -o /capture/qualification.png')
        (self.output/'qualification.png').write_bytes(self.remote('cat '+folder+'/qualification.png'))
        if [s1[10],s2[10],s3[10]]!=[65,153,2] or s3[5]!=1 or s3[4]!=2 or s3[7]!=0 or not s1[11]<s2[11]<=s3[11]:raise ValueError(f'Native qualification failed: {s0}, {s1}, {s2}, {s3}')
        dump(path,dict(utc=utc(),ssh_host=self.o.host,game_hashes=GAME_HASHES,image_id=self.image_id,actor_sha256=ACTOR,observer_sha256=OBSERVER,policy_sha256=SFT,
                       preflight_only=True,signed_native_states={'paused':s0,'left_shoot':s1,'up_right_focus_shoot':s2,'bomb':s3}))
        self.release(name);print('QUALIFIED',self.o.host,flush=True)
    def credit(self,job):
        t,seed=job['ticks'],job['seed'];tag=f'held-sft{t}-{seed}';folder=self.output/tag;folder.mkdir(exist_ok=True)
        if (folder/'receipt.json').is_file():return
        ident=f'{seed}-{t}';name,remote_dir=self.boot(ident,t);start=utc()
        self.exec(name,shlex.join(['sh','/capture/lib/collect-credit.sh',tag,str(seed),'/capture/lib/sft.bin']),True)
        print('ACTIVE',self.o.host,tag,flush=True)
        path=remote_dir+'/'+tag
        for _ in range(570):
            status=self.remote(f'if test -s {path}/settled.png; then echo ready; else cat {path}/actor.log 2>/dev/null || true; fi').decode().strip()
            if status=='ready':break
            if status and 'terminal=0' in status:raise RuntimeError('Technical endpoint '+tag)
            time.sleep(4)
        else:raise TimeoutError(tag)
        for _ in range(70):
            state=self.state(remote_dir)
            if state and state[2]==1 and state[4]==-1 and state[7]==0:break
            time.sleep(.5)
        else:raise ValueError('Unverified native ranking '+tag)
        self.exec(name,'scrot -o /capture/'+tag+'/ranking.png && cp /capture/telemetry.bin /capture/'+tag+'/final-stock.bin')
        members=['steps.bin','cadence.bin','events.tsv','actor.log','initial.png','ranking.png','final-stock.bin','settled.png']
        blob=self.remote('cd '+path+' && tar -czf - '+' '.join(members),timeout=120)
        with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as archive:
            for n in members:(folder/n).write_bytes(archive.extractfile(n).read())
        receipt=pack(folder,'held',seed,self.o.checkpoint,folder/'rollout.npz')
        cadence=np.fromfile(folder/'cadence.bin','<i4').reshape(-1,8)
        if len(cadence)!=receipt['steps'] or not np.all(cadence[:,0]==t):raise ValueError('Cadence row pairing mismatch')
        gap=cadence[:,2]-cadence[:,1];regular=cadence[:,3]==0
        if not regular.any() or not np.all(gap[regular]>=t):raise ValueError('Advancing action gap violates assigned horizon')
        np.savez_compressed(folder/'cadence.npz',target=cadence[:,0],native_tick_at_action=cadence[:,1],native_tick_at_release=cadence[:,2],
                            stop_reason=cadence[:,3],stage_at_action=cadence[:,4],stage_at_release=cadence[:,5],pause_at_release=cadence[:,6],hold_us=cadence[:,7])
        receipt.update(host=self.o.host,build='TH13 official Trial 0.01a Normal Reimu',arm='sft'+str(t),seed=seed,start_utc=start,finish_utc=utc(),native_ranking_tick=state[11],
                       native_terminal_pause_state=state[2],ranking_spares=state[4],ranking_active_lives=0,ranking_bombs=state[5],ranking_display_score=state[6]*10,
                       requested_native_ticks=t,advancing_stops=int(regular.sum()),exact_target_stops=int((regular&(gap==t)).sum()),overshoot_stops=int((regular&(gap>t)).sum()),
                       guard_stops=int((cadence[:,3]==1).sum()),transition_stops=int((cadence[:,3]==2).sum()),fatal_stops=int((cadence[:,3]==3).sum()),
                       advancing_gap_quantiles=np.percentile(gap[regular],[0,50,95,100]).tolist(),median_hold_ms=float(np.median(cadence[:,7])/1000),
                       cadence_sha256=digest(folder/'cadence.npz'),original_sha256=GAME_HASHES[0],ranking_photo='ranking.png')
        dump(folder/'receipt.json',receipt)
        self.release(name);print('COMPLETE',self.o.host,tag,receipt['steps'],receipt['native_ranking_tick'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--host',required=True);p.add_argument('--known-hosts',type=Path)
    p.add_argument('--remote-root',default='/tmp/toho-th13-cadence-261125');p.add_argument('--assets',default='/tmp/toho-th13-261125/assets/app')
    p.add_argument('--launcher',default='/tmp/toho-th13-lm-261125/src/launch.sh');p.add_argument('--image',default='toho-trial:endeavor-native-v6')
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--registration',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,choices=range(1,7),default=4);p.add_argument('--qualification-only',action='store_true')
    opts=p.parse_args();pool=Pool(opts);pool.gate();pool.qualification()
    if not opts.qualification_only:
        plan=json.loads(opts.registration.read_bytes())
        if plan['policy_sha256']!=SFT:raise ValueError('Unverified registration checkpoint')
        with ThreadPoolExecutor(opts.workers) as workers:list(workers.map(pool.credit,plan['allocation']))
