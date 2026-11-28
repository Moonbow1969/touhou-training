"""Verify raw held TH08 Normal outcomes against their signed physical clock and source traces."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.audit_th08_live_tick4 import game_word,read_records
from scripts.report_th08_stock_exposure import terminal_stage_evidence,postprompt_stage1_tick


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def inspect(raw,receipt):
    for filename,property_name in (("gates.tsv","input_gates_sha256"),("clock.tsv","stage_clock_trace_sha256"),
                                  ("timing.jsonl","timing_sha256"),("terminal.png","terminal_photo_sha256")):
        if digest(raw/filename)!=receipt[property_name]:
            raise ValueError(f"The native source changed: {filename}")
    source=json.loads((raw/"receipt.json").read_bytes())
    if (source["role"]!="held" or receipt["role"]!="held" or source["seed"]!=receipt["seed"] or
            source["policy_sha256"]!=receipt["policy_sha256"] or receipt["outcome"]!="game_over" or receipt["continues"]!=0):
        raise ValueError("The native role, controller or terminal differs")
    endpoint=terminal_stage_evidence(raw,receipt)
    observations=[json.loads(s) for s in (raw/"timing.jsonl").read_text().splitlines()]
    active=[r for r in observations if r["phase"]=="policy"]
    gates=np.genfromtxt(raw/"gates.tsv",names=True,encoding="ascii",dtype=None,ndmin=1)
    words=read_records(raw)
    if len(active)!=len(gates) or any(int(g["x11_mask"])!=r["mask"] for g,r in zip(gates,active)):
        raise ValueError("Independently logged per-decision controls differ")
    stale=[]
    for j in range(1,len(active)-1):
        earlier,current,nxt=gates[j-1:j+2]
        start=int(current["apply_target"])+1;stop=int(nxt["apply_target"])+1
        stage=int(current["stage"])
        if stage!=nxt["stage"] or stop-start!=4:continue
        actual=words[stage][start:stop]
        previous=game_word(int(earlier["x11_mask"]));requested=game_word(int(current["x11_mask"]))
        if len(actual)==4 and requested!=previous and (actual==previous).all():
            stale.append(dict(decision=j,stage=stage,first_tick=start,last_tick=stop-1,requested=requested,preceding=previous,
                              original_slots=actual.tolist()))
    first=int(gates[0]["apply_target"])+1
    terminal=None
    if endpoint["stage"]==1:
        if endpoint["native_terminal_stage"]==1:terminal=receipt["terminal_tick"]
        else:
            proof=postprompt_stage1_tick(raw,receipt)
            if proof["verified"]:terminal=proof["first_native_stopped_tick"]
    # Rebuild the physical reserve ledger directly from pre-action HUD states and original Game Over.
    lives=[r["spares"] for r in active]+[observations[-1]["spares"]]
    losses=[j for j in range(len(active)) if lives[j+1]<lives[j]]
    if any(lives[j]-lives[j+1]!=1 for j in losses):
        raise ValueError("Original reserve decrease needs independent multi-spare review")
    final_idx=receipt["decisions"]-1
    positive=sum(active[j]["bombs"]>0 for j in [*losses,final_idx])
    return dict(seed=receipt["seed"],actor_sha256=receipt["policy_sha256"],stage=endpoint["stage"],stage2=endpoint["stage2"],
                first_tick=first,last_stage1_tick=terminal,stage1_at_risk_ticks=None if terminal is None else terminal-first+1,
                deaths=len(losses)+1,deaths_with_positive_seen_bomb=positive,terminal_stock=[observations[-1]["spares"],observations[-1]["bombs"]],
                middle_wholly_stale_commands=stale)


def verify(study):
    report_path=study/"results.json"
    if not report_path.exists():report_path=study/"partial-results.json"
    report=json.loads(report_path.read_bytes());binding=json.loads((study/"approved-release.json").read_bytes())
    if binding["registration_sha256"]!=digest(study/"held-registration.json"):
        raise ValueError("The registered physical comparison changed")
    rows=[]
    for record in report["by_original"]:
        seed=record["seed"];arm=record["arm"];folder=study/arm
        receipt=json.loads((folder/"sealed"/f"{seed}.json").read_bytes())
        found=inspect(folder/"raw"/str(seed),receipt)
        if (found["seed"]!=seed or found["actor_sha256"]!=binding["arms"][arm]["sha256"] or
                found["stage2"]!=record["stage2"] or found["terminal_stock"]!=[record["terminal_spares"],record["terminal_bombs"]] or
                found["deaths"]!=record["total_lost_lives"] or found["deaths_with_positive_seen_bomb"]!=record["losses_with_positive_pre_loss_bomb"] or
                found["middle_wholly_stale_commands"]!=record["middle_wholly_stale_commands"]):
            raise ValueError(f"Original controls, resource ledger or stage differ in {seed} {arm}")
        if found["stage"]==1 and found["stage1_at_risk_ticks"]!=record.get("stage1_at_risk_ticks"):
            raise ValueError("Original Stage 1 exposure differs")
        rows.append(dict(arm=arm,**found))
    return dict(checked_originals=len(rows),score_sha256=digest(report_path),records=rows)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study",type=Path,help="Held folder with raw original archives extracted into <arm>/raw/<seed>/")
    parser.add_argument("--out",type=Path)
    args=parser.parse_args();data=verify(args.study)
    output=json.dumps(data,indent=2)+"\n"
    if args.out:args.out.write_text(output,encoding="utf8")
    else:print(output,end="")
