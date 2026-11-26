"""Report observed vulnerable boss phases and stocks on complete original-game credits."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.checkpoint_storage import digest
from scripts.train_full_policy import publish_json


def read_credit(report_path, stage):
    data = json.loads(report_path.read_bytes())
    if data["status"] != "complete" or data["specification"]["task"]["version"] != "1.00b":
        raise ValueError("Need completed original TH14 1.00b story credits.")
    results = []
    for record in data["completed"]:
        credit = record["result"]
        if credit["native_continues"] or credit["outcome"] not in ("game_over", "one_credit_clear"):
            raise ValueError("Unverified terminal outcome.")
        entry = credit["transition_shard"]
        path = report_path.parent / entry["path"]
        if digest(path) != entry["sha256"]:
            raise ValueError("Changed gameplay trace.")
        with np.load(path, allow_pickle=False) as z:
            c = {name:i for i,name in enumerate(json.loads(str(z["metadata_json"]))["native_field_order"])}
            b, a, actions = (z[k] for k in ("native_before", "native_after", "action"))
            cols = [c[k] for k in ("stage", "boss_pointer", "boss_max_hp", "spell_id")]
            active = (b[:,c["stage"]] == stage) & (b[:,c["boss_pointer"]] > 0) & (b[:,c["boss_hp"]] > 0)
            active &= (b[:,c["snapshot_valid"]] == 1) & (b[:,c["boss_vulnerable"]] == 1)
            ids = np.flatnonzero(active)
            changes = np.any(np.diff(b[ids][:,cols], axis=0) != 0, axis=1) | (np.diff(ids) > 1)
            phases = []
            for ix in np.split(ids, np.flatnonzero(changes)+1):
                if not len(ix):
                    continue
                first = b[ix[0]]
                linked = (b[ix][:,cols] == a[ix][:,cols]).all(1) & (a[ix,c["snapshot_valid"]] == 1)
                linked &= (a[ix,c["boss_hp"]] >= 0) & z["four_tick_stage_transition"][ix].astype(bool)
                idx = ix[linked]
                hp = np.r_[b[ix,c["boss_hp"]], a[idx,c["boss_hp"]]]
                phases.append(dict(start_frame=int(first[c["stage_frame"]]), end_frame=int(b[ix[-1],c["stage_frame"]]),
                    max_hp=int(first[c["boss_max_hp"]]), spell=int(first[c["spell_id"]]),
                    start_hp=int(first[c["boss_hp"]]), lowest_linked_or_observed_hp=int(hp.min()),
                    decisions=len(ix), shoot_pressed=int(actions[ix,2].sum()), exact_linked_decisions=int(len(idx)),
                    confirmed_life_losses=int(z["native_life_losses"][ix].sum()),
                    actual_bombs_used=int(z["actual_bombs_consumed"][ix].sum())))
        results.append(dict(trial=record["trial"], stage=stage, outcome=credit["outcome"],
                            stage_entries={s["stage"]:s["entry"]["resources"] for s in credit["stage_records"]},
                            terminal=credit["native_terminal"], phases=phases, trace_sha256=entry["sha256"]))
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", type=int, choices=range(1,7), default=2)
    p.add_argument("--reports", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    rows = [row for path in args.reports for row in read_credit(path, args.stage)]
    publish_json(args.out, dict(credits=rows, observation="Contiguous vulnerable decisions. Phase disappearance alone cannot distinguish depletion, scripted transitions and timeouts."))
