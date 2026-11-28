"""Summarize descriptive original endpoints for every attempt, including primary exclusions."""
import csv
import hashlib
import json
from pathlib import Path
import sys


def extract(study):
    sys.path.insert(0,str(study.parent/"tools"))
    from seal import source_info
    cohort=json.loads((study/"completed-pairs.json").read_bytes())
    output=[]
    for pair in cohort["pairs"]:
        for arm in pair["order"]:
            seed=pair["seed"]
            folder=study/f"{seed}-{arm}"
            result=dict(seed=seed,arm=arm,primary_credit_eligible=pair["arms"][arm]["valid"],
                        primary_pair_eligible=pair["valid"])
            for name in ("finished-game.json","official-policy.rpy","ui-verified.json","settled-scorecard.png",
                         "native-input.csv","observer-timing.txt"):
                result.setdefault("source_sha256",{})[name]=hashlib.sha256((folder/name).read_bytes()).hexdigest()
            with (folder/"native-input.csv").open() as f:
                obs=list(csv.DictReader(f))
            stages={int(r["stage"]) for r in obs}
            last=obs[-1]
            ui=json.loads((folder/"ui-verified.json").read_bytes())
            finished=json.loads((folder/"finished-game.json").read_bytes())
            official_score=source_info(folder/"official-policy.rpy",max(stages))[2]
            result.update(stage1_ticks=max(int(r["tick"]) for r in obs if r["stage"]=="1"),
                          stage2=2 in stages,terminal=finished["outcome"],terminal_stock=[int(last["lives"]),int(last["bombs"])],
                          native_recorded_score=10*int(last["score"]),official_saved_score=official_score,
                          misses=int(last["misses"]),actual_bombs=int(last["bombs_used"]),continues=int(last["continues"]),
                          scorecard_ok=ui["scorecard"],rank=last["rank"],shot=last["shot"],
                          max_read_ms=int((folder/"observer-timing.txt").read_text().split("=")[1])/1e6)
            output.append(result)
    return output


if __name__=="__main__":
    study=Path(sys.argv[1]).resolve()
    (study/"all-original-endpoints.json").write_text(json.dumps(extract(study),indent=2)+"\n")
