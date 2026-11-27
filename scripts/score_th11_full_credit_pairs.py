"""Compare stock TH11 full credits by native HUD score, inventory and verified Stage 2."""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.collect_th10_full_action import save
from scripts.run_th10_camera_cohort import digest


def score(study, public):
    registration = json.loads((study / "camera-protocol.json").read_bytes())
    arms = [registration["reference_arm"], registration["candidate"]]
    seeds = registration["eval_seeds"]["11"]
    source = study / "th11-held-qualified-native.json"
    rows = json.loads(source.read_bytes())["credits"]
    assert len(rows) == 2 * len(seeds) and len(set(seeds)) == len(seeds)
    for r in rows:
        folder = study / "native11" / f"held-{r['arm']}-{r['seed']}"
        receipt_file = folder / "receipt.json"
        receipt = json.loads(receipt_file.read_bytes())
        assert digest(receipt_file) == r["receipt_sha256"] and receipt["role"] == "held"
        units = receipt["power_units_per_level"]
        assert units == 20
        r.update(host=receipt["host"], power_units_per_level=units,
                 settled_power=r["settled_power_units"] / units,
                 power_at_fatal=r["fatal_power_units"] / units)
    lookup = {(r["arm"], r["seed"]): r for r in rows}
    assert set(lookup) == {(a, s) for a in arms for s in seeds}
    ranks = registration["rank_by_seed"]["11"]
    measures = ["settled_score", "power_at_fatal", "settled_power", "effective_casts", "spares"]
    rng = np.random.default_rng(26711)
    results = {}
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.1), constrained_layout=True)
    for rank, title in [(2, "Hard"), (3, "Lunatic")]:
        subset = [s for s in seeds if ranks[str(s)] == rank]
        pairs = [[lookup[(arm, s)] for arm in arms] for s in subset]
        assert all(left["host"] == right["host"] for left, right in pairs)
        assert all(r["difficulty"] == rank and r["power_units_per_level"] == 20 for pair in pairs for r in pair)
        bootstrap = rng.integers(0, len(pairs), size=(20000, len(pairs)))
        counts = dict(n_pairs=len(pairs), paired_seeds=subset, arms=arms,
            stage2=[sum(pair[i]["stage2_reached"] for pair in pairs) for i in (0, 1)],
            first_stage2_resources=[[pair[i]["first_stage2"] for pair in pairs if pair[i]["stage2_reached"]] for i in (0, 1)],
            effective_casts=[sum(pair[i]["effective_casts"] for pair in pairs) for i in (0, 1)],
            input_matches=[sum(pair[i]["native_key_matches"] for pair in pairs) for i in (0, 1)],
            decisions=[sum(pair[i]["n"] for pair in pairs) for i in (0, 1)])
        for name in measures:
            vals = np.array([[r[name] for r in pair] for pair in pairs], dtype=float)
            changes = vals[:, 1] - vals[:, 0]
            counts[name + "_summary"] = dict(means=vals.mean(axis=0).tolist(), paired_mean=float(changes.mean()),
                paired_mean_ci95=np.quantile(changes[bootstrap].mean(axis=1), [.025, .975]).tolist())
        for ax, name, label in zip(axes[rank - 2], ["settled_score", "power_at_fatal", "effective_casts"],
                                    ["Settled HUD score (millions)", "Power at last loss", "Confirmed casts"]):
            vals = np.array([[r[name] for r in pair] for pair in pairs], dtype=float)
            if name == "settled_score":
                vals /= 1e6
            for i, pair in enumerate(vals):
                offset = (i - (len(vals) - 1) / 2) * .006
                ax.plot([offset, 1 + offset], pair, marker="o", markersize=3,
                        linewidth=.9, alpha=.65, color="#47778c")
            ax.plot([0, 1], vals.mean(axis=0), marker="D", linewidth=2, color="#202327")
            ax.set(xticks=[0, 1], xticklabels=arms, ylabel=label, title=title)
            ax.grid(axis="y", alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(-.15, 1.15)
        results[title.lower()] = counts
    fig.suptitle("TH11 full-stock original-game evaluation · Same-seed paired credits")
    figure = study / "th11-matched-original-outcomes.png"
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    report = study / "th11-matched-original-outcomes.json"
    save(report, dict(arms=arms, source_sha256=digest(source), protocol_sha256=digest(study / "camera-protocol.json"),
        score_unit="Settled native score multiplied by ten and verified against photographed HUD.",
        power_unit="Twenty native units per one HUD Power level.",
        uncertainty="Percentile bootstrap over whole same-seed pairs, 20,000 resamples stratified by difficulty.",
        ranks=results))
    target = public / study.name
    target.mkdir(parents=True, exist_ok=True)
    for file in (figure, report):
        shutil.copy2(file, target / file.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--public", type=Path, default=Path("results/2026-11-26-original-camera-training"))
    args = parser.parse_args()
    score(args.study, args.public)
