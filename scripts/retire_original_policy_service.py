"""Retire one explicitly owned, sealed-study inference container by its immutable ID."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def digest(file):
    with file.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def completed_roster(study):
    protocol = study / "camera-protocol.json"
    ledger = study / "evaluation/credits.json"
    summary = study / "evaluation/publication-summary.json"
    if not all(f.is_file() for f in (protocol, ledger, summary)):
        return None
    registered = json.loads(protocol.read_bytes())
    expected = {(int(title), seed, arm) for title, seeds in registered["eval_seeds"].items()
                for seed in seeds for arm in (registered["reference_arm"], registered["candidate"])}
    rows = json.loads(ledger.read_bytes())
    observed = {(int(r["title"]), r["seed"], r["arm"]) for r in rows}
    if observed != expected or len(rows) != len(expected) or any(not r["terminal"] for r in rows):
        raise ValueError("Published native endpoint roster differs from every registered arm")
    return {"study": study.name, "physical_credits": len(rows), "protocol_sha256": digest(protocol),
            "ledger_sha256": digest(ledger), "published_summary_sha256": digest(summary)}


def declared_consumers(data, title, arm):
    owners = []
    for file in sorted(data.glob("*/camera-protocol.json")):
        plan = json.loads(file.read_bytes())
        titles = {str(k) for key in ("train_seeds", "eval_seeds")
                  for k, seeds in plan.get(key, {}).items() if seeds}
        if str(title) in titles and arm in (plan.get("training_arm"), plan.get("reference_arm"), plan.get("candidate")):
            owners.append(file.parent.name)
    return owners


def retire(request, apply=False):
    service = request["service"]
    engine = request["engine"]
    study = request["study"]
    owners = request["registered_consumers"]
    if engine not in ("docker", "podman") or not re.fullmatch(r"[a-z][a-z0-9-]{3,120}", service):
        raise ValueError("Invalid declared service identity")
    if owners != [study]:
        return {"status":"shared-or-unowned", "retryable":False, "consumers":owners}
    gate = request["completion"]
    if gate["study"] != study or gate["physical_credits"] < 1 or any(
            not re.fullmatch("[0-9a-f]{64}", gate[k]) for k in ("protocol_sha256", "ledger_sha256", "published_summary_sha256")):
        raise ValueError("Incomplete original results provenance")
    sha = request["policy_sha256"]
    if not re.fullmatch("[0-9a-f]{64}", sha):
        raise ValueError("Invalid policy digest")
    def call(*args):
        return subprocess.run([engine,*args],capture_output=True,text=True,timeout=30)
    current = call("inspect",service)
    if current.returncode:
        if any(s in current.stderr.lower() for s in ("no such object", "no such container", "no such container or image")):
            return {"status":"already-absent", "retryable":False}
        return {"status":"inspect-failed", "retryable":True, "error":current.stderr[-360:]}
    info = json.loads(current.stdout)[0]
    cid = info["Id"]
    if not re.fullmatch("[0-9a-f]{64}",cid) or info["Name"].lstrip("/")!=service:
        raise ValueError("Immutable container ID/name mismatch")
    if request.get("container_ids") and cid not in request["container_ids"]:
        return {"status":"deployment-identity-changed","retryable":False,"id":cid}
    cfg=info["Config"]
    argv=cfg.get("Cmd") or []
    role=Path(argv[0]).name if argv else ""
    if (role not in ("camera_service.py","camera_service_v261126.py") or
            info["HostConfig"]["NetworkMode"]!="none" or "--sha" not in argv or
            argv[argv.index("--sha")+1]!=sha or "--model" not in argv or
            argv[argv.index("--model")+1]!=request["inside_model"]):
        return {"status":"service-identity-mismatch","retryable":False,"id":cid}
    sock=argv[argv.index("--socket-name")+1] if "--socket-name" in argv else "actor.sock"
    if sock!=request["socket_name"]:
        return {"status":"socket-identity-mismatch","retryable":False,"id":cid}
    if not info["State"]["Running"]:
        return {"status":"already-stopped","retryable":False,"id":cid}
    actual=call("exec",cid,"sha256sum",request["inside_model"])
    if actual.returncode or actual.stdout.split()[:1]!=[sha]:
        return {"status":"cannot-verify-mounted-checkpoint","retryable":True,"id":cid}
    sockets=call("exec",cid,"cat","/proc/net/unix")
    if sockets.returncode:
        return {"status":"cannot-verify-sockets","retryable":True,"id":cid}
    fields=[line.split() for line in sockets.stdout.splitlines()[1:]]
    listens=[r for r in fields if len(r)>=8 and r[3]=="00010000" and r[5]=="01" and Path(r[7]).name==sock]
    nonlisteners=[r for r in fields if len(r)<7 or r[5]!="01" or r[3]!="00010000"]
    if len(fields)!=1 or len(listens)!=1 or nonlisteners:
        return {"status":"unix-socket-busy-or-unknown","retryable":True,
                "id":cid,"matching_listeners":len(listens),"other_sockets":len(nonlisteners)}
    if not apply:
        return {"status":"eligible-idle","retryable":False,"id":cid}
    stopped=call("stop","-t","3",cid)
    if stopped.returncode:
        return {"status":"stop-failed","retryable":True,"id":cid,"error":stopped.stderr[-360:]}
    check=call("inspect","--format","{{.State.Running}}",cid)
    return {"status":"stopped" if check.returncode==0 and check.stdout.strip()=="false" else "verify-stop-failed",
            "retryable":not(check.returncode==0 and check.stdout.strip()=="false"),"id":cid}


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--request",type=Path,help="Audited request with completion, exclusive consumers, image command, model SHA and socket. Default stdin.")
    cli.add_argument("--apply",action="store_true")
    args=cli.parse_args()
    request=json.loads(args.request.read_bytes()) if args.request else json.load(sys.stdin)
    print(json.dumps(retire(request,args.apply),sort_keys=True))
