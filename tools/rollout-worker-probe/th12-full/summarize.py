"""Collect complete native TH12 credits and seal verified full-action learner inputs."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def summarize(raw, output, stage2_step=None):
    mode, seed, fingerprint, episode=(raw/"identity.txt").read_text().split()
    end, count, wall=(raw/"outcome.txt").read_text().split()
    if end != "game_over":
        raise ValueError("This full-credit collector requires original terminal evidence.")
    with (raw/"decisions.tsv").open(encoding="ascii") as f:
        rows=list(csv.DictReader(f,delimiter="\t"))
    n=len(rows)
    if int(count)!=n or n<1 or rows[-1]["status"] not in ("gameover","gameover_retry","score_ranking"):
        raise ValueError("Missing original terminal frame or action rows.")
    action=np.array([[int(r[k]) for k in ("move","focus","shoot","bomb")] for r in rows],np.int8)
    logp=np.array([[float(r["logp_"+k]) for k in ("move","focus","shoot","bomb")] for r in rows],np.float32)
    value=np.array([float(r["value"]) for r in rows],np.float32)
    features=np.fromfile(raw/"features.f32","<f4").reshape(n,110)
    now=np.array([int(r["observed_ns"]) for r in rows],np.int64)
    onset=np.array([int(r["onset_ns"]) for r in rows],np.int64)
    keys=np.array([int(r["keys"]) for r in rows],np.int16)
    sensed=np.array([int(r["verified"]) for r in rows],np.int16)
    if not (np.array_equal(keys,sensed) and np.isfinite(features).all() and np.isfinite(logp).all()
            and np.isfinite(value).all() and (now>=onset).all() and np.all(np.diff(now)>0)):
        raise ValueError("Signed behavior or action clock validation failed.")
    valid=np.array([r['status']=='play' for r in rows])
    observed=np.array([[int(r['spares']),int(r['bombs'])] for r in rows],np.int16)
    inventory=[2,2];events=[]
    rewards=np.zeros((n,5),np.float32)
    rewards[:,0]=.001*(now-onset)/1e9*valid
    for t in range(n-2):
        # Three consecutive native active observations stop transient HUD changes becoming events.
        if not valid[t:t+3].all():continue
        observed_l,observed_b=map(int,observed[t])
        old_l,old_b=inventory
        if observed_l>=0 and np.all(observed[t:t+3,0]==observed_l) and old_l!=observed_l:
            rewards[t,1]=observed_l-old_l
            inventory[0]=observed_l
            events.append(dict(step=t,event="spares",prior=old_l,current=observed_l))
        if observed_b>=0 and np.all(observed[t:t+3,1]==observed_b) and old_b!=observed_b:
            if (old_b>observed_b and np.all(observed[max(0,t-2):min(n,t+4),0]==old_l)
                    and np.any(action[max(0,t-5):t+1,3])):
                rewards[t,2]=-.03*(old_b-observed_b)
            inventory[1]=observed_b
            events.append(dict(step=t,event="bombs",prior=old_b,current=observed_b,cost=float(rewards[t,2])))
    stage2_spares = None
    stage2_bombs = None
    stage2_stock_source = None
    if stage2_step is not None:
        if not 0<=stage2_step<n:raise ValueError("Invalid verified stage transition.")
        rewards[stage2_step,3]=2
        # Report the HUD at the verified action, supporting each resource independently.
        # A loading fade or a changing reserve cannot certify the inventory at entry.
        scan = slice(stage2_step, stage2_step + 3)
        if len(valid[scan]) == 3 and valid[scan].all():
            spares, bombs = map(int, observed[stage2_step])
            if spares >= 0 and np.all(observed[scan, 0] == spares):
                stage2_spares = spares
            if bombs >= 0 and np.all(observed[scan, 1] == bombs):
                stage2_bombs = bombs
        stage2_stock_source = ("native HUD at the verified Stage 2 action, confirmed by the next two active observations"
                               if stage2_spares is not None or stage2_bombs is not None
                               else "entry HUD obscured or unstable")
    rewards[-1,4]=-1
    terminated=np.zeros(n,dtype=bool); terminated[-1]=True
    if inventory[0]!=0:raise ValueError("Native spare stock before Game Over requires review.")
    output.mkdir(parents=True,exist_ok=True)
    record=output/(episode+".npz")
    np.savez_compressed(record,features=features,action=action,logp=logp,value=value,reward=rewards.sum(axis=1),
                        terminated=terminated,reward_parts=rewards,keys=keys,observed_keys=sensed,
                        stock_hud=observed,onset_ns=onset,observed_ns=now)
    frames=sorted((raw/'evidence').glob('*.jpg'))
    if (raw/'terminal-limit.json').exists():
        cut=json.loads((raw/'terminal-limit.json').read_bytes())["original_terminal_ns"]/1e9
        clocks={name:float(when) for when,name in (raw/"evidence/clock.txt").read_text().splitlines() and
                (r.split() for r in (raw/"evidence/clock.txt").read_text().splitlines())}
        frames=[p for p in frames if clocks[p.name]<=cut+1.8]
    sheet=Image.new('RGB',(720,((len(frames[::4])+5)//6)*108+26),'#14222b');dr=ImageDraw.Draw(sheet)
    dr.text((7,7),f'{episode}: original one-credit progression, every fourth camera observation',fill='white')
    for i,p in enumerate(frames[::4]):
        thumb=Image.open(p).convert('RGB')
        thumb.thumbnail((116,86));x=i%6*120;y=i//6*108+26
        sheet.paste(thumb,(x,y));dr.text((x+2,y+87),p.stem,fill='white')
    sheet.save(output/(episode+'-sheet.jpg'),quality=75)
    terminal_image = raw / "terminal-at-control.ppm"
    terminal_hash = hashlib.sha256(terminal_image.read_bytes()).hexdigest() if terminal_image.exists() else None
    report=dict(title=12,game="TH12 UFO official trial 0.02a",episode=episode,seed=int(seed),mode=mode,
                lane=episode.split('-')[-1],policy_sha256=fingerprint,actions=n,seconds=float(wall),
                native_outcome=end,stage2_reached=stage2_step is not None,stage2_step=stage2_step,
                native_terminal_state=rows[-1]["status"],terminal_observation_sha256=terminal_hash,
                stage2_remaining_spare_lives=stage2_spares,stage2_remaining_bombs=stage2_bombs,
                stage2_stock_source=stage2_stock_source,
                stage2_review="Review the time-stamped original color progression and terminal screen for each credit.",
                remaining_spare_lives=inventory[0],remaining_bombs=inventory[1],initial_spares=2,initial_bombs=2,
                event_history=events,move_counts=np.bincount(action[:,0],minlength=9).tolist(),
                key_positive_counts=action[:,1:].sum(axis=0).tolist(),reward_parts=rewards.sum(axis=0).tolist(),
                reward_parts_order=["valid survival","spare stock differential","actual same-life bomb spending",
                                    "verified stage progression","game over"],
                trajectory=record.name,trajectory_sha256=hashlib.sha256(record.read_bytes()).hexdigest(),
                native_masks_verified=n,actor_seed_only=True,game_rng_controlled=False,
                screenshot_count=len(frames),median_control_ms=float(np.median(np.diff(onset))/1e6) if n>1 else None)
    (output/(episode+".json")).write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return report
