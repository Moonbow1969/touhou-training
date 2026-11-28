"""Keep all attempted original credits if the fixed technical replacement budget is exhausted."""
import fcntl
import json
from pathlib import Path
import sys


def seal(study):
    with (study/"pair-dispatch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        destination=study/"completed-pairs.json"
        if destination.exists():
            return
        plan=json.loads((study/"held-protocol.json").read_bytes())
        paths=list(study.glob("pair-*.json"))
        pairs=[json.loads(p.read_bytes()) for p in sorted(paths, key=lambda p:p.stat().st_mtime_ns)]
        assert set(plan["primary_seeds"]).issubset(p["seed"] for p in pairs)
        reserves=sum(p["seed"] in plan["technical_reserves"] for p in pairs)
        failures={arm:sum(not p["arms"][arm]["valid"] for p in pairs) for arm in plan["models"]}
        assert reserves == len(plan["technical_reserves"]) or max(failures.values()) > 2
        valid=[p for p in pairs if p["valid"]]
        actual={arm:sum(p["arms"][arm]["ticks"] for p in valid) for arm in plan["models"]}
        result=dict(pairs=pairs,valid_pairs=len(valid),exposure=actual,
            technical_failures_per_arm=failures,technical_reserves_used=reserves,
            closure="The preregistered technical replacement limit stopped further enrollment.")
        data=destination.with_suffix(".tmp")
        data.write_text(json.dumps(result,indent=2)+"\n")
        data.replace(destination)


if __name__=="__main__":
    seal(Path(sys.argv[1]))
