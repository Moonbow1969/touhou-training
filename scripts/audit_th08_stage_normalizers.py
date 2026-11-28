"""Bind tentative TH08 0.02 stage scales to the five registered Normal Ghost TRAIN players."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from statistics import median

import numpy as np

from replays.inspect_th08_trial002_sources import inspect, stage_words

ORIGINAL = "3a6b4a0de24f916e669cf642ffd4f7c22ada1d319d9ce8ca8494fe96a2f5f5c1"
STATE = np.dtype([("tick","<u4"),("stage","<u4"),("keys","<u4"),("prev","<u4"),
                  ("cursor","<u4"),("origin","<u4"),("flags","<u4"),("mode","<u4"),("camera","u1",(112,96))])
TRAIN = {"pawa", "tak.", "katiwo", "wai eymo", "yoshi"}


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def derive(root, replay_dir):
    corpus = root / "th08-trial002-human"
    protocol = root / "th08-normal-temporal/protocol.json"
    plan = json.loads(protocol.read_bytes())
    assert plan["rank"] == "Normal" and plan["team"] == "Ghost"
    # Never open held replay bytes, cameras or qualification records.
    inputs = [root / v for v in plan["archives"]["train"]]
    assert len(inputs) == len(set(inputs)) == 15
    rows = []
    for path in inputs:
        metadata = path.with_suffix(".json")
        q = json.loads(metadata.read_bytes())
        author, stage = q["player_group"], q["stage"]
        assert author in TRAIN and q["split"] == "train" and q["difficulty"] == "Normal"
        assert q["game_version"] == "0.02_" and q["team"] == "Ghost Team" and q["original_executable_sha256"] == ORIGINAL
        source = replay_dir / Path(q["source"]).name
        raw = source.read_bytes()
        desc = inspect(raw)
        assert desc == q["audit"]["source"] and desc["replay_sha256"] == q["replay_sha256"]
        replay = source.stem
        words = stage_words(raw)[stage]
        proof = corpus / "original-playback-runs" / replay
        orig = json.loads((proof / "provenance.json").read_bytes())
        assert orig["replay_sha256"] == q["replay_sha256"] and orig["qualified_stages"] == orig["observed_stages"] == 3
        assert orig["split"] == "train" and orig["player"] == author
        for file in ("stage-selector.png", "normal-speed.png", "replay-finish.png", "sources.sha256"):
            assert sha(proof/file) == orig["menu_and_playback_evidence"][file]
        src = corpus / "verified-source-frames" / replay / f"stage-{stage}"
        state = np.memmap(src/"frames.bin",dtype=STATE,mode="r")
        assert sha(src/"frames.bin") == q["audit"]["raw_camera_sha256"]
        assert np.all(state["mode"] == 1) and np.all(state["stage"] == stage)
        assert np.all(state["flags"] & 4) and not np.any(state["flags"] & 512)
        t = state["tick"].astype("int64")
        assert np.all(np.diff(t) > 0) and np.all(t%4 == 0) and np.all(t < len(words))
        assert np.array_equal(state["cursor"], state["origin"] + 0x24 + 2*t)
        # The original's replay controller supplies the preceding stream word.
        active = np.uint32(0x00F7)
        assert np.array_equal(state["keys"] & active, words[t-1] & active)
        assert np.array_equal(state["prev"] & active, words[t-2] & active)
        sealed = (src / "observer.tsv").read_text().strip().splitlines()[-1]
        m = re.fullmatch(r"sealed\s+(\d+)\s+last_tick\s+(\d+)\s+racing\s+\d+\s+cursor_mismatch\s+0",sealed)
        assert m and (int(m[1]),int(m[2])) == (len(state),int(t[-1]))
        assert q["audit"]["source_ticks"] == len(words)
        rows.append(dict(author=author,original_replay_name=desc["native_pilot"],stage=stage,
                         mode=1,original_team=desc["native_team"],native_header_final_stage=desc["native_stage"],
                         source_url=q["source"],source_sha256=q["replay_sha256"],first_observed_tick=int(t[0]),
                         last_observed_tick=int(t[-1]),sampled_camera_count=len(t),dummy_inclusive_key_words=len(words),
                         source_words_minus_last_tick=len(words)-int(t[-1]),last_8_source_words=[int(k)for k in words[-8:]],
                         original_camera_sha256=sha(src/"frames.bin"),observer_sha256=sha(src/"observer.tsv"),
                         qualification_json_sha256=sha(metadata),run_provenance_sha256=sha(proof/"provenance.json"),
                         return_to_replay_menu_sha256=sha(proof/"replay-finish.png")))
    assert {(r["author"],r["stage"]) for r in rows} == {(p,s) for p in TRAIN for s in (1,2,3)}
    for r in rows:
        following = next((q for q in rows if q["author"]==r["author"] and q["stage"]==r["stage"]+1),None)
        r["next_stage_first_verified_tick"] = following["first_observed_tick"] if following else None
        r["stage_completed"] = True if following else None
        r["exact_last_playable_tick"] = None
    groups = {str(s):[r["last_observed_tick"] for r in rows if r["stage"]==s] for s in (1,2,3)}
    return dict(scope="TH08 0.02 Normal Ghost; five frozen TRAIN authors; original replay, no new gameplay",
        source_protocol="th08-normal-temporal/protocol.json",source_protocol_sha256=sha(protocol),
        access_scope="Only TRAIN paths from the frozen Normal roster; no held source or camera opened.",
        game_executable_sha256=ORIGINAL,stage_ticks_candidate={s:int(median(groups[s])) for s in ("1","2")},
        endpoint_statistic="Median of final actually observed, source-key-matched, playback-active stage ticks for stages confirmed to have completed by later original stage ordinal. Conservative normalization proxy, not a transition-at-this-exact-tick claim.",
        stage_3=dict(normalizer=None,sampled_tail_median=int(median(groups["3"])),ending_status="Unknown. All original USER sections report Stage 3; final screenshots show the Replay menu. No Stage 4 or unambiguous original clear/ending witness in these data."),
        timing_contract="Native camera accepted ticks divisible by four with flags & 4 and !(flags & 512), source cursor=origin+0x24+2*t, keys[t-1]/previous[t-2]. Dummy-inclusive stream length and trailing zeros/held keys are separate observations. No adjustment for unsampled ticks and no inference of a terminal from key release.",
        rows=rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("root","replays","out"):
        parser.add_argument("--"+arg,type=Path,required=True)
    args=parser.parse_args()
    doc=derive(args.root,args.replays)
    args.out.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
