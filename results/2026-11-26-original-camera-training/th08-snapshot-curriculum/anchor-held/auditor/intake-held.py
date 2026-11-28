"""Independently seal restored-game held sources, then acknowledge their transfer."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time


def load(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".pending")
    temp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def link(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        os.link(source, dest)
    assert sha(source) == sha(dest)


def evaluate(args):
    working = args.registration.parent
    first_transfer = args.incoming / "80965501" / "compact-transfer.done"
    first_failure = first_transfer.with_name("technical-failure.json")
    while not (args.registration.exists() and (first_transfer.exists() or first_failure.exists())):
        time.sleep(2)
    reg = load(args.registration)
    assert [r["seed"] for r in reg["rows"]] == list(range(80965501, 80965541))
    assert all(r["role"] == "held" for r in reg["rows"])
    assert reg["control_filters"]["maximum_total_exclusions"] == 4
    assert reg["replace_attempts"] is False
    source_files = {"lambda1-parent": args.parent, "mixed-ppo": args.candidate}
    source_hash = {arm: sha(path) for arm, path in source_files.items()}
    assert len(set(source_hash.values())) == 2
    for row in reg["rows"]:
        assert row["policy_sha256"] == source_hash[row["arm"]]
    checker = Path(__file__).with_name("verify-held-fragment.py")
    spec = importlib.util.spec_from_file_location("held_original_evidence", checker)
    verify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verify)
    verify.reg = reg
    verify.registration = args.registration
    verify.prefix = verify.npz(args.prefix.read_bytes())
    verify.rank_template = verify.np.load(args.glyph, allow_pickle=False)
    verify.torch.set_num_threads(1)
    models = {arm: verify.torch.jit.load(str(path), map_location="cpu").eval() for arm,path in source_files.items()}
    attempts = []
    consumed_disk = 0
    for row in reg["rows"]:
        seed, arm = row["seed"], row["arm"]
        case = args.incoming / str(seed)
        marker = working / "attempts" / f"{seed}.json"
        compact, shard = case / "compact.tar.gz", case / "fragment.npz"
        if marker.is_file():
            attempt = load(marker)
        else:
            transferred = case / "compact-transfer.done"
            while not (transferred.is_file() and compact.is_file() and shard.is_file()) and not (case / "technical-failure.json").is_file():
                time.sleep(2)
            if (case / "technical-failure.json").is_file():
                failure = case / "technical-failure.json"
                evidence = working / "evidence" / f"{seed}-technical-failure.json"
                link(failure,evidence)
                attempt = dict(row,qualified=False,physical_endpoint_verified=False,
                               evidence=evidence.relative_to(working).as_posix(),evidence_sha256=sha(evidence),
                               failure=load(failure))
            else:
                verify.model,verify.actor = source_files[arm],models[arm]
                result = verify.accept(seed,compact,shard)
                n = verify.npz(shard.read_bytes())
                inside = (n["spares"] == n["next_spares"]) & (n["bombs"] > n["next_bombs"])
                spent = int((n["bombs"][inside] - n["next_bombs"][inside]).sum())
                drop = verify.np.flatnonzero(n["next_spares"] < n["spares"])
                loss = None
                if len(drop):
                    index = int(drop[0])
                    loss = {
                        "policy_decision": index,
                        "native_observation_tick_before": int(n["native_tick"][index]),
                        "native_observation_tick_after": int(n["native_next_tick"][index]),
                        "spares_before": int(n["spares"][index]),
                        "spares_after": int(n["next_spares"][index]),
                        "bombs_before": int(n["bombs"][index]),
                        "bombs_after": int(n["next_bombs"][index]),
                    }
                native = n["native_word_slots"]
                boundary = str(result["outcome"])
                physical = boundary in ("game_over","life_loss_cutoff","native_tick_cutoff")
                a,b = working/"evidence"/f"{seed}-compact.tar.gz",working/"shards"/f"{seed}.npz"
                link(compact,a)
                link(shard,b)
                receipt = working/"receipts"/f"{seed}.json"
                publish(receipt,result)
                attempt = dict(row,qualified=result["qualified"],physical_endpoint_verified=physical,
                           endpoint=boundary,ticks=result["elapsed_native_ticks"],first_stock=result["first_stock"],
                           end_stock=result["end_stock"],decisions=result["decisions"],consumed=result["action_masks_with_native_consumption"],
                           same_life_observed_bomb_units_spent=spent,exact_windows=result["control_evidence"]["exact_windows"],
                           native_observed_bomb_spend_events=int(inside.sum()),first_observed_spare_loss=loss,
                           policy_x_requested_decisions=int(n["action"][:,3].sum()),
                           native_x_active_slots=int((((native>=0)&((native&2)!=0))).sum()),
                           comparable_windows=result["control_evidence"]["total_windows"],right_censored=(boundary=="native_tick_cutoff"),
                           file=b.relative_to(working).as_posix(),sha256=sha(b),evidence=a.relative_to(working).as_posix(),
                           evidence_sha256=sha(a),receipt=receipt.relative_to(working).as_posix(),receipt_sha256=sha(receipt))
            publish(marker,attempt)
        attempts.append(attempt)
        if attempt.get("file"):
            consumed_disk += sum((working/attempt[key]).stat().st_size for key in ("file","evidence"))
        n_bad = sum(not item["qualified"] for item in attempts)
        exact = sum(item.get("exact_windows",0) for item in attempts)
        total = sum(item.get("comparable_windows",0) for item in attempts)
        # Preserve every physically completed outcome, including controls outside the strict gate.
        proceed = n_bad <= 4 and consumed_disk <= reg["compact_evidence_limit_bytes"]
        status = dict(seed=seed,arm=arm,physical=len(attempts),qualified=attempt["qualified"],
                      valid=attempt["qualified"],accepted=len(attempts)-n_bad,excluded=n_bad,
                      exact_windows=attempt.get("exact_windows",0),comparable_windows=attempt.get("comparable_windows",0),
                      pooled_exact_windows=exact,pooled_comparable_windows=total,processed_bytes=consumed_disk)
        status["continue"] = proceed
        publish(working/"intake-progress.json",status)
        publish(case/"intake-ack.json",status)
        print(json.dumps(status),flush=True)
        if not proceed:
            break
    publish(working/"original-outcomes.json",dict(attempts=attempts,complete=len(attempts)==40,
                                                 registration_sha256=sha(args.registration),source_sha256=source_hash))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    for key in ("registration","incoming","parent","candidate","prefix","glyph"):
        parser.add_argument("--"+key,type=Path,required=True)
    evaluate(parser.parse_args())
