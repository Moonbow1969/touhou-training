"""Exercise corruption, duplicated rows, missing physical action proof and original score settlement."""
import csv
import json
from pathlib import Path
import struct
import sys
import tempfile
import zlib
from record_proof import official_score, verify_archive
from verify_record import verify


def events(folder):
    data=(folder/"events.bin").read_bytes()
    i=0
    while i<len(data):
        _,lane,n=struct.unpack_from("<III",data,i)
        yield lane,data[i+12:i+12+n]
        i+=16+n


def fixture(source,path,rows):
    omitted={"events.bin","archive.sealed","native-input.csv","applied.csv","verified.json"}
    for f in source.iterdir():
        if f.is_file() and f.name not in omitted:
            (path/f.name).symlink_to(f.resolve())
    table=[bytearray((source/name).read_bytes().splitlines(keepends=True)[0])
           for name in ("native-input.csv","applied.csv")]
    data=bytearray()
    for seq,(lane,payload) in enumerate(rows):
        record=struct.pack("<III",seq,lane,len(payload))+payload
        data.extend(record+struct.pack("<I",zlib.crc32(record)))
        table[lane].extend(payload)
    (path/"events.bin").write_bytes(data)
    (path/"archive.sealed").write_text(f"{len(rows)}\n")
    for name,payload in zip(("native-input.csv","applied.csv"),table):
        (path/name).write_bytes(payload)


def run(source,score_lag_source):
    allrows=list(events(source))
    with (source/"controls.csv").open() as f:
        action=list(csv.DictReader(f))[20]
    tick=int(action["tick"])
    clean=verify_archive(source)
    assert clean["stable_duplicate_rows"]>0
    dropped=[r for r in allrows if r[0] or not tick<int(r[1].split(b",")[1])<=tick+4]
    with tempfile.TemporaryDirectory(dir=source.parent) as temp:
        path=Path(temp)
        fixture(source,path,dropped)
        new=verify(path,True)
        assert not new["passed"] and int(action["seq"]) in new["stale"]
    with tempfile.TemporaryDirectory(dir=source.parent) as temp:
        path=Path(temp)
        wrong=list(allrows)
        i=next(i for i,r in enumerate(wrong) if r[0]==0 and int(r[1].split(b",")[1])==tick)
        pieces=wrong[i][1].decode().rstrip().split(",")
        pieces[-1]=str(tick+1)
        wrong.insert(i+1,(0,(",".join(pieces)+"\n").encode()))
        fixture(source,path,wrong)
        try:
            verify_archive(path)
        except ValueError as exc:
            assert "Inconsistent native clock" in str(exc)
        else:raise AssertionError("Torn duplicate was accepted")
    with tempfile.TemporaryDirectory(dir=source.parent) as temp:
        path=Path(temp)
        fixture(source,path,allrows)
        damaged=bytearray((path/"events.bin").read_bytes())
        damaged[12]^=1
        (path/"events.bin").write_bytes(damaged)
        try:
            verify_archive(path)
        except ValueError as exc:
            assert "CRC" in str(exc)
        else:raise AssertionError("Later corruption was accepted")
    with tempfile.TemporaryDirectory(dir=source.parent) as temp:
        path=Path(temp)
        for name in ("settled-scorecard.png","settled-scorecard-repeat.png"):
            (path/name).symlink_to((score_lag_source/"settled-scorecard.png").resolve())
        score=official_score(path,625830,625510)
        assert score["score_confirmed"] and score["settlement_delta"]==320
        assert not official_score(path,625820,625510)["score_confirmed"]
    return dict(clean_duplicate_rows_verified=clean["stable_duplicate_rows"],crc_corruption_rejected=True,
        semantically_torn_duplicate_with_valid_crc_rejected=True, lost_whole_native_witness_rejected=True,
        lost_window_tick=tick,official_rpy_unchanged=True,prior_score_lag=score,
        score_fixture_scope="One archived scorecard reused twice for deterministic regression; live engineering requires two distinct capture commands.",
        screenshot_and_rpy_disagreement_rejected_for_score=True)


if __name__=="__main__":
    report=run(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
    Path(sys.argv[3]).write_text(json.dumps(report,indent=2)+"\n")
