"""Audit the independent Accelerator original TH13 camera pilot before assigning training credits."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from scripts.collect_th10_full_action import save


def process_cost(records, wall, cpu):
    t=np.array([r[wall] for r in records]);usage=np.array([r[cpu] for r in records])
    scale=1e-6 if cpu=='cpu_usec' else 1.
    assert len(t)>2 and (np.diff(t)>0).all()
    return dict(samples=len(t),measured_s=float(t[-1]-t[0]),mean_cpu_cores=float((usage[-1]-usage[0])*scale/(t[-1]-t[0])),
                peak_rss_or_cgroup_bytes=max(r['memory_bytes'] for r in records))


def audit(root):
    root=root.resolve();out=root/'engineering/th13-accelerator'
    deployment=json.loads((out/'deployment.json').read_bytes())
    folder=out/'held-accelerator-qual-ppo4-967301'
    record=json.loads((folder/'receipt.json').read_bytes())
    data=np.fromfile(folder/'steps.bin','<f4').reshape(-1,143)
    assert data.shape[0]==record['steps'] and record['x_key_match']==record['steps']
    assert record['continues']==0 and record['terminal'] and record['remaining_lives']==-1
    assert record['native_terminal_pause_state']==1 and not data[:-1,120].any() and data[-1,120]==1
    assert data[0,[125,127,129,141]].tolist()==[1,2,2,1]
    assert record['native_key_match']/record['steps']>=.98
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    policy=root/'camera-ppo4/policy.ts'
    assert record['policy_sha256']==deployment['policy_sha256']==sha(policy)
    assert record['sha256']==sha(folder/'rollout.npz')
    with np.load(folder/'rollout.npz') as z:
        raw=z['frames'];actions=z['action'];logged=z['logp'];critic=z['value']
    assert np.array_equal(raw,np.fromfile(folder/'observations.u8','u1').reshape(-1,112,96))
    torch.set_num_threads(2);model=torch.jit.load(str(policy),map_location='cpu').eval()
    with torch.inference_mode():
        gray=torch.from_numpy(raw).float()/255;delta=torch.zeros_like(gray);delta[1:]=(gray[1:]-gray[:-1]).abs()
        features=torch.stack((gray,delta),1);selected=[];values=[]
        for start in range(0,len(raw),24):
            logits=model(features[start:start+24])
            a=torch.from_numpy(actions[start:start+24]).long()
            selected.append(torch.stack([h.log_softmax(1).gather(1,a[:,k:k+1])[:,0] for k,h in enumerate(logits[:4])],1))
            values.append(logits[4].reshape(-1))
        pn=float(np.abs(torch.cat(selected).numpy()-logged).max());vn=float(np.abs(torch.cat(values).numpy()-critic).max())
        assert max(pn,vn)<3e-4,(pn,vn)
    cgroup=[json.loads(line) for line in (out/'runtime.jsonl').read_text().splitlines()]
    sidecar=[json.loads(line) for line in (out/'camera-service-usage.jsonl').read_text().splitlines()]
    # Stage-local clock resets at stage entry. Only aligned within-stage advances are in the numerator.
    before,after=data[:,134],data[:,135]
    good=(data[:,125]==data[:,126])&(after>=before)
    steps=np.maximum(np.diff(np.r_[before[0],after]),0)
    native_ticks=int(steps[good].sum())
    game_cpu=process_cost(cgroup,'monotonic_s','cpu_usec')
    inference_cpu=process_cost(sidecar,'host_monotonic_s','cpu_s')
    report=dict(qualified=True,host='accelerator',engineering_seed=record['seed'],game_sha256=record['game_sha256'],
        policy_sha256=record['policy_sha256'],native_opening=dict(stage=1,difficulty='Normal',spare_lives=2,bombs=2,score=0,continues=0),
        terminal=record['outcome'],terminal_spares=record['remaining_lives'],terminal_bombs=record['remaining_bombs'],
        stage2=record['stage2'],seconds=record['wall_seconds'],decisions=record['steps'],native_ticks=native_ticks,
        native_stage_ticks_per_wall_second=native_ticks/record['wall_seconds'],decisions_per_second=record['steps']/record['wall_seconds'],
        median_action_hold_ms=record['median_hold_ms'],native_keys_matching=record['native_key_match'],x11_keys_matching=record['x_key_match'],
        movement_direction_counts=np.bincount(actions[:,0],minlength=9).tolist(),focus=record['focus_steps'],shoot=record['shoot_steps'],
        requested_bomb_decisions=record['bomb_key_steps'],effective_bombs=record['effective_bombs'],confirmed_losses=record['confirmed_losses'],
        full_observation_recomputed_max_logp_delta=pn,full_observation_recomputed_max_value_delta=vn,
        wine_container_cgroup=game_cpu,camera_sidecar=inference_cpu,
        cpu_scope='Cgroup sampling includes private Wine setup and shutdown. The host TorchScript inference sidecar is sampled independently over its observed active window; the two means have different time denominators.',
        raw_files_sha256={k:sha(folder/k) for k in ('steps.bin','events.tsv','initial.png','ranking.png','final-stock.bin','rollout.npz','receipt.json')})
    save(out/'qualification.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    audit(p.parse_args().root)
