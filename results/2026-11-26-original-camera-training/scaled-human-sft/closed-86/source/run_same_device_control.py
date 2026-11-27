"""Sequential same-GPU rerun of the old data after the new 86-source milestones finish."""
import json
import os
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import time
root=Path(__file__).resolve().parent;results=root/"results";out=results/"same-device-old64";out.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_bytes())
def digest(p):return sha256(p.read_bytes()).hexdigest()
def run(name,*args):subprocess.run([sys.executable,"-u","-m","scripts."+name,*map(str,args)],cwd=root/"code",check=True)
while not (results/"completed.json").is_file():time.sleep(10)
f=root/"same-device-old64-protocol.json";plan=read(f)
assert digest(f)=="0f782ec859aae5d1c8cc40756eced6d95ba440bbb8874390a4e48c9aefb58f6b"
assert digest(root/"code/scripts/train_human_camera_policy96.py")==plan["same_learning_source_sha256"]
assert digest(root/"models/encoder.pt")==plan["same_encoder_sha256"]
assert not {r["credited_author_group"] for r in plan["train"]}&{r["credited_author_group"] for r in plan["held"]}
for r in plan["train"]+plan["held"]:
    p=root/"data"/r["path"];assert digest(p)==r["archive_sha256"] and digest(p.with_suffix(".json"))==r["metadata_sha256"]
import torch
assert torch.cuda.is_available() and "5090" in torch.cuda.get_device_name()
train=[root/"data"/r["path"] for r in plan["train"]];held=[root/"data"/r["path"] for r in plan["held"]]
run("train_human_camera_policy96","--encoder",root/"models/encoder.pt","--reference",root/"models/ppo3.bin","--train",*train,"--held",*held,"--out",out/"training","--steps",44149,
    "--natural-movement-frequency","--balance-titles","--player-location-weight",.15,"--cpu-replay","--spatial-attention")
model=out/"matched-44149/policy";report=read(model/"report.json")
assert report["steps"]==44149 and sum(r["decisions"] for r in report["train"])==235457
run("compare_human_camera_policies","--models",model,"--held",*held,"--out",out/"held-movement")
run("score_human_bomb_onsets","--models",model,"--held",*held,"--out",out/"held-onsets.json")
run("score_replay_data_scaling","--reference",model,"--candidate",results/"matched-44149/policy","--out",out/"matched-data-comparison","--labels","77 credited train authors","97 credited train authors")
(out/"completed.json").write_text(json.dumps(dict(step=44149,presentations=44149*64,held_credited_authors=6,
     earlier_data_5090_sha256=digest(model/"policy.ts"),new_data_5090_sha256=digest(results/"matched-44149/policy/policy.ts"),
     comparison="matched-data-comparison")),encoding="utf8")
