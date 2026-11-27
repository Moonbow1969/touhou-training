"""Recompute historical normalized TH12 stage outcomes from hash-matched original-game records."""
import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path

from scripts.checkpoint_storage import digest
from scripts.score_native_full_actions import summarize
from scripts.score_original_camera_run import verified_th12_stage


def replace_json(path, old, new):
    if old == new:
        return
    if json.loads(path.read_bytes()) != old:
        raise RuntimeError(f"Result changed during migration: {path}")
    temporary = path.with_suffix(".th12-stage.tmp")
    temporary.write_text(json.dumps(new, indent=2) + "\n", encoding="utf8")
    temporary.replace(path)


def migrate(raw_root, result_roots):
    receipts = {}
    for directory in raw_root.rglob("*native12*"):
        if not directory.is_dir() or not (directory / "raw").is_dir():
            continue
        for file in (directory / "credits").glob("*.json"):
            record = json.loads(file.read_bytes())
            key = record.get("trajectory_sha256")
            if not key or not record.get("episode"):
                continue
            raw = directory / "raw" / record["episode"]
            if raw.is_dir():
                receipts.setdefault(key, []).append((file, raw, record))

    @lru_cache(maxsize=None)
    def certify(trace_hash, source_hash, endpoint_hash, policy_hash, seed):
        matches = []
        for source, raw, record in receipts[trace_hash]:
            if record["seed"] != seed or record["policy_sha256"] != policy_hash:
                continue
            if digest(source) != source_hash:
                continue
            terminal = raw / "terminal-at-control.ppm"
            if not terminal.is_file():
                terminal = raw / "endpoint.png"
            if digest(terminal) != endpoint_hash or digest(source.parent / record["trajectory"]) != trace_hash:
                continue
            matches.append(verified_th12_stage(raw))
        if len(set(matches)) != 1:
            raise ValueError(f"No unique verified result for trace={trace_hash} seed={seed} sources={len(matches)}")
        return matches[0]

    def repair(row):
        if row.get("title") != 12:
            return row
        reached, evidence = certify(row["original_trace_sha256"], row["native_log_sha256"],
                                    row["original_endpoint_sha256"], row["policy_sha256"], row["seed"])
        return dict(row, stage2_reached=reached, stage2_evidence_sha256=evidence)

    bundles = []
    movie_count = 0
    records = {}
    paths = []
    for root in result_roots:
        for path in sorted(root.rglob("credits.json")):
            old = json.loads(path.read_bytes())
            if not isinstance(old, list) or not all(isinstance(r, dict) for r in old):
                continue
            th12 = [row for row in old if row.get("title") == 12 and row.get("original_trace_sha256")]
            if not th12:
                continue
            updated = [repair(row) for row in old]
            replace_json(path, old, updated)
            paths.append(str(path))
            rows = [row for row in updated if row.get("title") == 12]
            records.update({r["original_trace_sha256"]:r["stage2_reached"] for r in rows})
            arm_names = {r["arm"] for r in rows}
            assert len(arm_names) == 2
            summaries = 0
            for other in sorted(path.parent.glob("*.json")):
                if other == path:
                    continue
                data = json.loads(other.read_bytes())
                if not isinstance(data,dict):
                    continue
                if isinstance(data.get("sources"),list) and any(
                        isinstance(v,dict) and v.get("title")==12 and v.get("original_trace_sha256") for v in data["sources"]):
                    replace_json(other,data,dict(data,sources=[repair(r) for r in data["sources"]]))
                    movie_count += 1
                    paths.append(str(other))
                if "12" not in data or not isinstance(data["12"],dict) or "pairs" not in data["12"]:
                    continue
                baseline = next(arm for arm in arm_names if arm in data["12"])
                candidate = next(arm for arm in arm_names if arm != baseline)
                seeds = sorted({r["seed"] for r in rows})
                protocol = {"reference_arm":baseline, "eval_seeds":{"12":seeds}}
                fixed = summarize(rows, protocol, candidate)[12]
                replace_json(other,data,dict(data,**{"12":fixed}))
                summaries += 1
                paths.append(str(other))
            if summaries != 1:
                raise ValueError(f"Expected one paired TH12 summary: {path}")
            bundles.append(dict(root=str(root),directory=str(path.parent.relative_to(root)),
                                rows=len(rows),stage1=sum(r["stage2_reached"] is False for r in rows),
                                stage2=sum(r["stage2_reached"] is True for r in rows),
                                unknown=sum(r["stage2_reached"] is None for r in rows)))
    return dict(bundles=bundles, video_source_manifests=movie_count, unique_originals=len(records),
                unique_stage1=sum(v is False for v in records.values()),unique_stage2=sum(v is True for v in records.values()),
                unique_unknown=sum(v is None for v in records.values()),files=paths)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--originals",type=Path,required=True)
    parser.add_argument("--results",nargs="+",type=Path,required=True)
    parser.add_argument("--receipt",type=Path,required=True)
    args=parser.parse_args()
    result=migrate(args.originals,args.results)
    args.receipt.write_text(json.dumps(result,indent=2)+"\n")
    print({k:v for k,v in result.items() if k not in ("files","bundles")},len(result["bundles"]))
