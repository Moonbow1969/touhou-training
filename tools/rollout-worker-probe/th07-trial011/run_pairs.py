"""Run a fixed four-tick TH07 original pair from an already-qualified immutable installation."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE=Path("/var/tmp/moonbow-th07-trial-261127")
STUDY=BASE/"locator-bc/native"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(path,doc):
    tmp=path.with_suffix(".new");tmp.write_text(json.dumps(doc,indent=2)+"\n");tmp.replace(path)
def load(path):return json.loads(path.read_text())

def verify(folder):
    subprocess.run([sys.executable,str(STUDY/"tool/verify_credit.py"),str(folder),"--base",str(BASE),"--study",str(STUDY)],check=True)
    return load(folder/"verified.json")

def qualify():
    reports=[]
    for lane,arm in enumerate(["action-xy","action"]):
        seed=plan["qualification_seeds"][lane]
        folder=STUDY/f"engineering-v5-{seed}-{arm}"
        deadline=time.monotonic()+1000
        while not (folder/"receipt.pending-visual.json").is_file():
            if time.monotonic()>deadline:raise RuntimeError(f"Engineering credit did not seal: {folder}")
            time.sleep(3)
        r=verify(folder)
        assert r["passed"] and r["correct_native_tick_fraction"]==1
        reports.append(dict(path=folder.name,report_sha=sha(folder/"verified.json"),exact=r["exact_fraction"],ticks=r["stage1_native_ticks"]))
    write(STUDY/"qualification.json",dict(protocol_sha256=sha(STUDY/"protocol.json"),passed=True,reports=reports,
          recording_alignment="Qualification independently verifies saved original word[t+1] equals settled live native[t].",
          artifacts={n:sha(STUDY/"tool"/n) for n in ("tick4_actor","native_locator_credit.py","verify_credit.py","continue-mask.bin")}))

def pair(seed,ordinal):
    lane=plan["seed_lane"][str(seed)]
    first=plan["first_arm_sequence"][ordinal%len(plan["first_arm_sequence"])]
    arms=[first,"action" if first=="action-xy" else "action-xy"]
    results={}
    for arm in arms:
        folder=STUDY/f"held-{seed}-{arm}"
        try:
            if not folder.exists():
                subprocess.run([sys.executable,str(STUDY/"run_credit.py"),"--role","held","--seed",str(seed),"--arm",arm,"--lane",str(lane)],check=True)
            if not (folder/"receipt.pending-visual.json").is_file():raise ValueError("Missing original terminal")
            r=verify(folder)
            results[arm]={"passed":r["passed"],"path":folder.name,"technical":None if r["passed"] else "registered credibility checks failed"}
        except Exception as exc:
            results[arm]={"passed":False,"path":folder.name,"technical":str(exc)}
    record={"seed":seed,"lane":lane,"order":arms,"arms":results,"valid":all(v["passed"]for v in results.values())}
    write(STUDY/f"pair-{seed}.json",record)
    return record

def run():
    if not (STUDY/"qualification.json").exists():qualify()
    gate=load(STUDY/"qualification.json")
    assert gate["passed"] and gate["protocol_sha256"]==sha(STUDY/"protocol.json")
    for n,h in gate["artifacts"].items():assert sha(STUDY/"tool"/n)==h
    all_pairs=[]
    first=plan["first_held_seed"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
        for offset in (0,2,4):
            futures=[pool.submit(pair,first+j,j) for j in (offset,offset+1)]
            all_pairs.extend(f.result()for f in futures)
    def valid():return [p for p in all_pairs if p["valid"]]
    def exposure():
        return {a:sum(load(STUDY/r["arms"][a]["path"]/"verified.json")["stage1_native_ticks"]for r in valid())for a in ("action","action-xy")}
    def invalid_counts():
        return {a:sum(not r["arms"][a]["passed"]for r in all_pairs)for a in ("action","action-xy")}
    reserves=iter(plan["reserve_seeds"])
    def replace():
        missing=len(all_pairs)-len(valid())
        used=sum(r["seed"]in plan["reserve_seeds"]for r in all_pairs)
        while missing>used:
            if max(invalid_counts().values())>2:raise RuntimeError("technical limit")
            seed=next(reserves,None)
            if seed is None:raise RuntimeError("two replacement pairs exhausted")
            all_pairs.append(pair(seed,len(all_pairs)))
            used+=1
            missing=len(all_pairs)-len(valid())
    replace()
    j=6
    while min(exposure().values())<30000 and len(valid())<12:
        all_pairs.append(pair(first+j,j));j+=1
        replace()
    scores={}
    for a in ("action","action-xy"):
        rows=[load(STUDY/r["arms"][a]["path"]/"verified.json")for r in valid()]
        deaths=sum(v["stage1_deaths"]for v in rows); ticks=sum(v["stage1_native_ticks"]for v in rows)
        scores[a]=dict(deaths=deaths,stage1_ticks=ticks,stage1_deaths_per_10000_ticks=deaths*10000/ticks,
                       mean_stage1_native_ticks=ticks/len(rows),stage2=sum(v["stage2"]for v in rows),number=len(rows))
    rr=scores["action-xy"]["stage1_deaths_per_10000_ticks"]/scores["action"]["stage1_deaths_per_10000_ticks"]
    progress=scores["action-xy"]["mean_stage1_native_ticks"]>=scores["action"]["mean_stage1_native_ticks"]
    write(STUDY/"complete.json",dict(all_pairs=all_pairs,valid=len(valid()),scores=scores,rate_ratio=rr,
         minimum_exposure_met=min(exposure().values())>=30000,promotion=rr<=0.75 and progress and min(exposure().values())>=30000,
         stop_intervention=rr>=1.0,technical_failures=invalid_counts(),native_rng="Uncontrolled. Descriptive exposure-normalized rates."))

if __name__=="__main__":
    plan=load(STUDY/"protocol.json")
    # Single dispatcher per namespace, lock released automatically at process exit.
    import fcntl
    with (STUDY/"dispatcher.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run()
