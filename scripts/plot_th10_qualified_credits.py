"""Summarize sealed TH10 original full-credit cohorts with qualified stage and stock reads."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.collect_th10_full_action import save
from scripts.run_th10_camera_cohort import digest


EXPERIMENTS = [
    ("v2-online-corrected-stage", "14-player CNN · Normal", "SFT", "PPO"),
    ("attention-online-corrected-stage", "Spatial attention · Normal", "SFT", "PPO"),
    ("easy-curriculum-corrected-stage", "Difficulty curriculum · Easy", "v2 PPO", "curriculum PPO"),
    ("power-loss", "Power lost on death · Easy", "curriculum PPO", "Power-loss PPO"),
]


def render(root, out):
    rows = []
    for folder, label, ref, alt in EXPERIMENTS:
        native = root / folder / "native10"
        receipts = native / "qualified-stage-receipts.json"
        result = native / "paired-stage-verified.json"
        if not receipts.is_file() or not result.is_file():
            continue
        report = json.loads(result.read_bytes())
        (left, a), (right, b) = report["group"].items()
        qualified = json.loads(receipts.read_bytes())
        assert a["stage2_sensor_validated"] and b["stage2_sensor_validated"]
        assert len(qualified) == a["n"] + b["n"]
        contrast = report["paired_versus_" + left][right]
        rows.append(dict(study=folder, label=label, reference=ref, candidate=alt, n=a["n"],
            stage2=[a["stage2_reached"], b["stage2_reached"]], first_stage2_resources=[a["stage2_resources"],b["stage2_resources"]],
            zero_spares_terminal=[a["final_spare_life_counts"], b["final_spare_life_counts"]],
            time_s=[a["mean_wall_survival_s"], b["mean_wall_survival_s"]],
            score=[a["mean_raw_score"], b["mean_raw_score"]],
            power=[a["mean_final_power"], b["mean_final_power"]],
            confirmed_strikes=[a["confirmed_bomb_spends"], b["confirmed_bomb_spends"]],
            difference_s=contrast["mean_survival_delta_s"],
            ci=contrast["paired_resampled_95pct_interval_for_mean_s"],
            receipts_sha256=digest(receipts)))
    if not rows:
        raise ValueError("No sealed qualified original cohorts")
    out.mkdir(parents=True, exist_ok=True)
    save(out / "summary.json", dict(title="TH10 official 0.02a trial", qualified_register="0x472b40",
         cohorts=rows, scope="Each row is an independent original-game assay. Within a row the two arms share controller seeds and physical host. Game RNG is uncontrolled. Spares -1 is the engine value after the last playable life."))
    fig, (ax, resources) = plt.subplots(1, 2, figsize=(10.5, 2.3 + .76*len(rows)), gridspec_kw={"width_ratios":[1.1,1]}, layout="constrained")
    for i, row in enumerate(rows):
        low, high = row["ci"]
        ax.hlines(i, low, high, color="#426780", lw=2)
        ax.scatter(row["difference_s"], i, color="#284b63", s=45, zorder=3)
    labels = [r["label"]+"\n"+r["reference"]+" → "+r["candidate"] for r in rows]
    ax.set_yticks(range(len(rows)), labels, fontsize=9)
    ax.set_ylim(len(rows)-.45, -.8); ax.axvline(0, color="#919ca5", ls="--", lw=.9)
    ax.set_xlabel("Change in mean survival (wall seconds)"); ax.grid(axis="x", alpha=.15)
    resources.set_axis_off()
    cells=[[" → ".join(f'{v}/{r["n"]}' for v in r["stage2"]),
            " → ".join(map(str,r["confirmed_strikes"])),
            " → ".join(f"{v:.2f}" for v in r["power"])] for r in rows]
    resources.set_xlim(0,3);resources.set_ylim(ax.get_ylim())
    for col, heading in enumerate(["Reached Stage 2","Total strikes","Final Power"]):
        resources.text(col+.5,-.6,heading,fontsize=9,weight="bold",ha="center",va="center")
    for i, values in enumerate(cells):
        if i%2==0:resources.axhspan(i-.40,i+.40,color="#edf3f6",zorder=0)
        for j,value in enumerate(values):
            resources.text(j+.5,i,value,fontsize=10,va="center",ha="center")
    fig.suptitle("TH10 original full credits · 12 matched controller seeds per policy", fontsize=14)
    fig.supxlabel("Intervals are paired 95% seed bootstrap intervals. Terminal Power pays for Spirit Strikes.", fontsize=9)
    ax.spines[["right","top"]].set_visible(False)
    fig.savefig(out/"paired-outcomes.png",dpi=160);plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True);parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();render(args.root,args.out)
