"""Apply the independent redundant-read audit to every original credit in the frozen roster."""
import csv
import json
from pathlib import Path
import sys
from reconcile_duplicate import adjudicate


def score(root):
    cohort=json.loads((root/"completed-pairs.json").read_bytes())
    plan=json.loads((root/"held-protocol.json").read_bytes())
    rows=[]
    for pair in cohort["pairs"]:
        for arm in pair["order"]:
            name=f"{pair['seed']}-{arm}"
            folder=root/name
            row=dict(seed=pair["seed"], arm=arm, strict_primary_valid=pair["arms"][arm]["valid"])
            try:
                audit=adjudicate(folder)
                (root/f"sensitivity-{name}.json").write_text(json.dumps(audit,indent=2)+"\n")
                q=audit["independent_verification"]
                row.update(eligible=audit["source_mask_gate"], duplicate_rows=audit["excluded_redundant_zero_based_rows"],
                           ticks=q["stage1_native_ticks"], deaths=q["stage1_deaths"], stage2=q["stage2"],
                           bomb=q["actual_bombs"], stock=q["terminal_stock"], exact=q["exact_fraction"], stale=q["stale"])
                with (folder/"controls.csv").open() as f:
                    keys=[int(r["requested"]) for r in csv.DictReader(f)]
                row.update(focus=sum(bool(w & 4) for w in keys), decisions=len(keys))
            except Exception as exc:
                row.update(eligible=False,reason=repr(exc))
            rows.append(row)
    ids=sorted({r["seed"] for r in rows if r["eligible"]} - {r["seed"] for r in rows if not r["eligible"]})
    arms={}
    for arm in plan["models"]:
        ok=[r for r in rows if r["seed"] in ids and r["arm"]==arm]
        ticks=sum(r["ticks"] for r in ok)
        deaths=sum(r["deaths"] for r in ok)
        arms[arm]=dict(n=len(ok),ticks=ticks,deaths=deaths,rate_per_10000=deaths*10000/ticks if ticks else None,
                      mean_stage1_ticks=ticks/len(ok) if ok else None,stage2=sum(r["stage2"] for r in ok),
                      actual_bombs=sum(r["bomb"] for r in ok),terminal_stock=[r["stock"] for r in ok],
                      focus_decisions=sum(r["focus"] for r in ok),decisions=sum(r["decisions"] for r in ok))
    p,c=arms.values()
    return dict(scope="All physically attempted credits. Sensitivity only. The registered primary score uses the original verifier and its own exposure-based enrollment.",
                included_seeds=ids,rows=rows,summary=arms,
                rate_ratio=c["rate_per_10000"]/p["rate_per_10000"] if p["rate_per_10000"] else None)


if __name__=="__main__":
    root=Path(sys.argv[1])
    (root/"sensitivity-summary.json").write_text(json.dumps(score(root), indent=2)+"\n")
