from pathlib import Path
import json,subprocess,sys,time
from engineering import lane,study
host=sys.argv[1]
plan=json.loads((study/"protocol.json").read_bytes())
order=[x for x in plan["lanes"] if x["host"]==host]
before=study/(host+"-results.json")
while not before.is_file() or len(json.loads(before.read_bytes()))<len(order):time.sleep(3)
rows=[]
for q in order:
    case=study/f"{q['seed']}-parent"
    if (case/"verified.json").exists():continue
    control=["ready.json","controls.csv","applied.csv","policy-trace.npz","native-input.csv","finished-game.json","events.bin","full-opening.png"]
    if any((case/k).exists() for k in control):
        rows.append(dict(seed=q["seed"],disposition="Actual decision/game evidence already exists; no replay."))
        continue
    assert (case/"capture-error.txt").exists()
    ev=study/("setup-before-original-"+case.name)
    assert not ev.exists() and case.parent==study
    case.rename(ev)
    try:rows.append(lane(q))
    except Exception as exc:
        import traceback
        (case/"second-setup-error.txt").write_text(traceback.format_exc())
        rows.append(dict(seed=q["seed"],qualified=False,error=repr(exc)))
    (study/(host+"-prestart-recovery.json")).write_text(json.dumps(rows,indent=2)+"\n")
