from pathlib import Path
import json,subprocess,sys,traceback
from engineering import lane,study
name=sys.argv[1]
protocol=json.loads((study/"protocol.json").read_bytes())
rows=[]
for q in protocol["lanes"]:
    if q["host"]!=name:continue
    assert q["role"]=="train" and q["native_rank"]==1
    try:rows.append(lane(q))
    except Exception as e:
        case=study/f"{q['seed']}-parent";case.mkdir(exist_ok=True)
        (case/"capture-error.txt").write_text(traceback.format_exc())
        rows.append(dict(seed=q["seed"],qualified=False,error=repr(e)))
        container=case/"actual-container"
        if container.is_file():
            c=container.read_text().strip();p=subprocess.run(["docker","inspect",c],capture_output=True)
            if p.returncode==0:
                assert any(m["Destination"]=="/capture" and Path(m["Source"])==case.resolve() for m in json.loads(p.stdout)[0]["Mounts"])
                subprocess.run(["docker","stop","-t","2",c],check=True)
    (study/(name+"-results.json")).write_text(json.dumps(rows,indent=2)+"\n")
