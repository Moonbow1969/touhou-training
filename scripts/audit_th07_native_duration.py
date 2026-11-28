"""Check duration-aware PPO targets by grouping already executed identical TH07 masks."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.checkpoint_storage import digest
from scripts.th07_native_duration import actor_eligible, duration_gae, native_discount, shape_progress


def identities(rng):
    tolerance = 0.
    for _ in range(50):
        ticks = rng.integers(1, 10, size=50)
        discounts = native_discount(ticks)
        phi = rng.integers(0, 8, size=len(ticks) + 1)
        prefix = np.r_[1., np.cumprod(discounts[:-1])]
        total = prefix @ shape_progress(np.zeros(len(ticks)), phi, discounts)
        expected = np.prod(discounts) * phi[-1] - phi[0]
        tolerance = max(tolerance, abs(total-expected))
    requested = np.array([5, 20, 5])
    executed = np.array([5]*4 + [20]*8 + [20, 5, 5, 5])
    np.testing.assert_array_equal(actor_eligible(requested, executed, [4, 8, 4]), [True, True, False])
    if tolerance > 2e-12:
        raise AssertionError(f"Potential failed to telescope on actual durations ({tolerance})")
    return float(tolerance)


def audit(source, rng):
    receipt = json.loads(source.with_suffix(".json").read_bytes())
    if receipt["sha256"] != digest(source) or receipt["qualified"] is not True:
        raise ValueError(f"Original transition source changed or failed control requirements: {source}")
    with np.load(source, allow_pickle=False) as data:
        ticks = data["native_ticks_under_command"]
        values = data["value"].astype(np.float64)
        action, stage, words = (data[name] for name in ("action", "native_stage", "native_words"))
        phi = 1. + np.r_[data["spares"], data["next_spares"][-1]].astype(np.float64)
    if receipt["outcome"] not in ("game_over", "practice_stage_clear") or np.any(stage != 1):
        raise ValueError("Expected a true single-stage original endpoint for this audit")
    discount = native_discount(ticks)
    base = ticks.astype(np.float64) / 10694
    if receipt["outcome"] == "game_over":
        phi[-1] = 0.
    else:
        base[-1] += 1.
    same_action = np.all(action[:-1] == action[1:], axis=1)
    same_native = np.all(words[:-1] == words[1:], axis=1)
    pure = np.all(words == words[:, :1], axis=1) & (ticks == 4) & (words[:, 0] >= 0)
    candidates = np.flatnonzero(pure[:-1] & pure[1:] & same_action & same_native)
    # Keep native terminal evidence as its own observed transition.
    candidates = candidates[(candidates > 0) & (candidates < len(values)-2)]
    chosen = set()
    for index in rng.permutation(candidates):
        if index-1 not in chosen and index+1 not in chosen:
            chosen.add(int(index))
        if len(chosen) >= max(1, round(.01*len(values))):
            break
    if not chosen:
        raise ValueError("Expected a nonempty retained prefix before the first grouped command")
    original = shape_progress(base, phi, discount)
    first, removed = min(chosen), {index + 1 for index in chosen}
    keep = np.array([index for index in range(len(values)) if index not in removed])
    grouped = np.array([base[i]+discount[i]*base[i+1] if i in chosen else base[i] for i in keep])
    duration = np.array([ticks[i]+ticks[i+1] if i in chosen else ticks[i] for i in keep])
    g = native_discount(duration)
    new_reward = shape_progress(grouped, phi[np.r_[keep, len(base)]], g)
    result = dict(seed=receipt["seed"], original_sha256=receipt["sha256"], decisions=len(values),
                  merged_pure_windows=len(chosen), prefix_decisions=first)
    for lam in (1., .985):
        reference = duration_gae(original, values, discount, lam)
        elapsed = duration_gae(new_reward, values[keep], g, lam)
        fixed = duration_gae(new_reward, values[keep], discount[keep], lam)
        prefix = duration_gae(original[:first], values[:first], discount[:first], lam, bootstrap=values[first])
        error, fixed_error, prefix_error = elapsed-reference[keep], fixed-reference[keep], prefix-reference[:first]
        result[str(lam)] = dict(smdp_mean=float(error.mean()), smdp_absolute_mean=float(np.abs(error).mean()),
                               smdp_max=float(np.abs(error).max()), fixed_absolute_mean=float(np.abs(fixed_error).mean()),
                               prefix_absolute_mean=float(np.abs(prefix_error).mean()))
    if result["1.0"]["smdp_max"] > 2e-11:
        raise AssertionError("Duration-aware returns changed on an equivalent grouping")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(21769523)
    output = dict(potential_identity_max_abs_error=identities(rng), sources=[audit(source, rng) for source in args.source])
    args.out.write_text(json.dumps(output, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(output))


if __name__ == "__main__":
    main()
