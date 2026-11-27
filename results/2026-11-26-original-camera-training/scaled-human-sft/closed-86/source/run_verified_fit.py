"""Verify every original training/held shard on Momentum, run the preregistered fit, then score both checkpoints."""
from datetime import datetime,timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

root=Path(__file__).resolve().parent;data=root/"data";code=root/"code";out=root/"results"
out.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_bytes())
def digest(p):
    h=sha256()
    with p.open("rb") as f:
        while chunk:=f.read(1<<20):h.update(chunk)
    return h.hexdigest()
def save(p,obj):
    temp=p.with_suffix(".writing");temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n","utf8");temp.replace(p)
started=datetime.now(timezone.utc).isoformat()
while not (root/"staging-complete").is_file():time.sleep(3)
plan_file=root/"protocol.json";plan=read(plan_file)
assert digest(plan_file)=="1cd1b8bdc7cb16ef3c684645d670b82d553c537aec7cb460c8e3f252b87482dc"
assert digest(code/"scripts/train_human_camera_policy96.py")==plan["instrumented_learning_source_sha256"]
assert digest(code/"scripts/train_human_camera_policy.py")==plan["unmodified_learning_source_sha256"]
assert digest(root/"models/encoder.pt")==plan["initialization_sha256"]
assert digest(root/"models/ppo3.bin")==plan["reference_ppo_sha256"]
assert digest(root/"models/previous-policy.ts")==plan["previous_actor_sha256"]
assert digest(data/"spatial-policy-study/protocol.json")==plan["base_protocol_sha256"]
assert digest(data/"human-train-incremental/registered-096-closed-credited-human-TRAIN.json")==plan["source_cohort_sha256"]
assert not {r["credited_author_group"] for r in plan["train"]}&{r["credited_author_group"] for r in plan["held"]}
assert not {r["original_header_group"] for r in plan["train"]}&{r["original_header_group"] for r in plan["held"]}
for row in plan["train"]+plan["held"]:
    p=data/row["path"];assert digest(p)==row["archive_sha256"] and digest(p.with_suffix(".json"))==row["metadata_sha256"]
import torch
assert torch.cuda.is_available() and "5090" in torch.cuda.get_device_name()
assert torch.ones(3,device="cuda").sum().item()==3
gpu=subprocess.check_output(["nvidia-smi","--query-gpu=index,name,driver_version,memory.total,memory.free,utilization.gpu","--format=csv,noheader"],text=True).strip()
save(out/"device-and-input-verification.json",dict(host="Momentum",role="dedicated GPU learner",device=gpu,torch_version=str(torch.__version__),cuda=torch.version.cuda,
    python=sys.version,registration_sha256=digest(plan_file),status="verified all 120 camera shards and sidecars plus encoder, reference and learner; CUDA kernel executed",
    train_shards=110,held_shards=10,train_credited=97,held_credited=6,registered_utc=started,verified_utc=datetime.now(timezone.utc).isoformat(),
    gpu_kernel_gemm_precision=torch.get_float32_matmul_precision(),disk_free_bytes=shutil.disk_usage(root).free))
print("INPUT AND CUDA VERIFIED",gpu,flush=True)
train=[str(data/r["path"]) for r in plan["train"]];held=[str(data/r["path"]) for r in plan["held"]]
def run(module,*args):
    subprocess.run([sys.executable,"-u","-m","scripts."+module,*map(str,args)],cwd=code,check=True)
run("train_human_camera_policy96","--encoder",root/"models/encoder.pt","--reference",root/"models/ppo3.bin","--train",*train,
    "--held",*held,"--out",out/"training","--steps",54485,"--natural-movement-frequency","--balance-titles","--player-location-weight",.15,"--cpu-replay","--spatial-attention")
prev=root/"previous";prev.mkdir(exist_ok=True)
for src,name in [("previous-policy.ts","policy.ts"),("previous-report.json","report.json")]:
    if not (prev/name).exists():(prev/name).hardlink_to(root/"models"/src)
for folder,n in [("matched-44149",44149),("corpus-54485",54485)]:
    arm=out/folder;model=arm/"policy";r=read(model/"report.json");counts=read(model/"observed-sampling.json")
    assert r["steps"]==n and counts["presentations"]==n*64 and sum(s["decisions"] for s in r["train"])==290583
    run("compare_human_camera_policies","--models",model,"--held",*held,"--out",arm/"held-movement")
    run("score_human_bomb_onsets","--models",model,"--held",*held,"--out",arm/"held-onsets.json")
    run("score_replay_data_scaling","--reference",prev,"--candidate",model,"--out",arm/"data-comparison","--labels","Earlier data at 44,149 updates",folder)
save(out/"completed.json",dict(registered_sha256=digest(plan_file),finish_utc=datetime.now(timezone.utc).isoformat(),
    checkpoints={k:digest(out/k/"policy/policy.ts") for k in ("matched-44149","corpus-54485")}))
