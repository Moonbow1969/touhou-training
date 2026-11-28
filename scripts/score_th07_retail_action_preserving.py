"""Check the frozen TH07 retail held-author and original-camera acceptance criteria."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def score(study, earlier):
    protocol_file = study / "protocol.json"
    protocol = load(protocol_file)
    result = load(study / "fit/result.json")
    previous = load(earlier / "fit/result.json")
    require(digest(protocol_file) == result["protocol_sha256"], "Unexpected fitted protocol")
    require(result["index_sha256"] == previous["index_sha256"] == protocol["index_sha256"], "Changed source rows")
    parent = result["models"]["parent"]
    require(parent["sha256"] == protocol["parent_sha256"] == previous["models"]["parent"]["sha256"], "Changed parent")
    gates = protocol["drift_gate"]
    rows = {}
    for seed in map(str, protocol["seeds"]):
        folder = study / "fit" / seed
        fitted = result["models"][seed]
        training = load(folder / "fit.json")
        baseline_epochs = load(earlier / "fit" / seed / "fit.json")["epochs"]
        sequence = [r["ordered_rows_sha256"] for r in training["epochs"]]
        require(len(sequence) == protocol["epochs"] and sequence == [r["ordered_rows_sha256"] for r in baseline_epochs],
                "Earlier comparison did not use identical minibatches")
        audit = load(folder / "original-drift.json")
        require(digest(folder / "policy.ts") == fitted["sha256"] == audit["sft_sha256"] == training["policy_sha256"],
                "Candidate differs from the fitted and audited model")
        require(audit["parent_sha256"] == parent["sha256"] and audit["source_sha256"] == protocol["engineering_cameras_archive_sha256"],
                "Camera source or reference differs")
        physical = {}
        for row in audit["per_source"]:
            if row["group"] != "bomb_stock_available":
                continue
            p, q = row["arms"]["parent"], row["arms"]["sft"]
            ratio = q["bomb"] / p["bomb"]
            delta = q["focus"] - p["focus"]
            physical[row["case"]] = dict(observations=row["observations"], bomb_ratio=ratio, focus_delta=delta,
                passed=(gates["bomb_probability_ratio_min"] <= ratio <= gates["bomb_probability_ratio_max"]
                        and abs(delta) <= gates["focus_positive_probability_absolute_difference_max"]))
        require(set(physical) == set(gates["cases"]), "Incomplete predefined camera cases")
        move = fitted["move"] - parent["move"]
        switch = fitted["switch"] - parent["switch"]
        held_pass = (move >= protocol["gate"]["mean_delta_move_accuracy_min"]
                     and switch >= protocol["gate"]["mean_delta_switch_move_accuracy_min"])
        authors = {k: dict(move_delta=v["accuracy"][0]-parent["authors"][k]["accuracy"][0],
                           switch_delta=v["switch_move_acc"]-parent["authors"][k]["switch_move_acc"],
                           windows=v["n"],switches=v["switches"]) for k,v in fitted["authors"].items()}
        rows[seed] = dict(sha256=fitted["sha256"], identical_ordered_epoch_hashes=sequence,
            move_delta=move, switch_delta=switch, localization_median_native_px=fitted["locator_median"],
            human_held_pass=held_pass, held_authors=authors, positive_stock_original=physical,
            all_gates_pass=held_pass and all(r["passed"] for r in physical.values()))
    fixed = str(protocol["candidate_seed"])
    return dict(protocol_sha256=digest(protocol_file), index_sha256=result["index_sha256"],
                original_archive_sha256=protocol["engineering_cameras_archive_sha256"],
                parent_sha256=parent["sha256"], seeds=rows, human_held_mean=result["mean"],
                registered_candidate=fixed, offline_eligible=rows[fixed]["all_gates_pass"],
                root_review_required_before_native=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--earlier", type=Path, required=True)
    args = parser.parse_args()
    (args.study / "acceptance.json").write_text(json.dumps(score(args.study, args.earlier), indent=2)+"\n", encoding="utf-8")
