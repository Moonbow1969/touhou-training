"""Derive and verify native TH10 Stage 1 progress from the independent clock log."""
import csv
import gzip
import hashlib
import io


def describe(data, rank, terminal_ms):
    rows=list(csv.DictReader(io.StringIO(data.decode("ascii"))))
    if len(rows)<2:
        raise ValueError("Missing native TH10 stage clock")
    utc=[int(row["utc_ms"]) for row in rows]
    stages=[int(row["stage"]) for row in rows]
    ticks=[int(row["stage_tick"]) for row in rows]
    lives=[int(row["lives"]) for row in rows]
    if utc != sorted(utc) or stages != sorted(stages) or stages[0]!=1 or min(ticks)<0 or set(stages)!=set(range(1,max(stages)+1)):
        raise ValueError("Invalid native TH10 clock order or stage range")
    if any(int(row["difficulty"])!=rank for row in rows) or min(lives[:-1])<0:
        raise ValueError("Invalid native TH10 clock difficulty or post-credit rows")
    if abs(utc[-1]-terminal_ms)>250:
        raise ValueError("Independent TH10 clock ended too far from original stock endpoint")
    last=stages.index(2)-1 if 2 in stages else len(stages)-1
    first_loss=next((i for i in range(1,len(rows)) if lives[i]<lives[i-1]),None)
    return dict(schema="th10-native-stage-clock-v1",stage_register="0x472b40",tick_register="0x472b4c",
                data_file="native-stage-clock.csv.gz",rows=len(rows),first_observed_tick=ticks[0],
                latest_observed_stage1_native_tick=ticks[last],last_stage1_row=last,
                last_stage1_utc_ms=utc[last],last_sidecar_spares=lives[-1],
                stock_terminal_ms=terminal_ms,sidecar_minus_stock_terminal_ms=utc[-1]-terminal_ms,
                first_life_loss_stage=int(stages[first_loss]) if first_loss is not None else None,
                first_life_loss_tick=int(ticks[first_loss]) if first_loss is not None else None)


def verify(folder,receipt):
    evidence=receipt.get("native_stage_clock")
    if not evidence:return None
    expected_source="7594fce626f19d3ba1cf731d4ac5a22ac73ca5cf1e8a0ee7aac6cf8a28247b0f"
    if (receipt["source"].get("stage_sensor")!="qualified-v2-clock" or evidence.get("source_sha256")!=expected_source
            or receipt["source"]["sources"].get("th10-stage-clock.c")!=expected_source):
        raise ValueError("Unqualified TH10 original clock source")
    clock=folder/"native-stage-clock.csv.gz"
    data=clock.read_bytes()
    if hashlib.sha256(data).hexdigest()!=evidence["data_sha256"]:
        raise ValueError("TH10 clock trace checksum mismatch")
    with gzip.open(folder/"native.csv.gz","rt") as f:
        last=None
        stock_stages=set()
        for last in csv.DictReader(f):
            stock_stages.add(int(last["stage"]))
    if last is None or int(last["lives"])>=0:
        raise ValueError("TH10 stock endpoint unavailable for clock verification")
    uncompressed=gzip.decompress(data)
    clock_stages={int(row["stage"]) for row in csv.DictReader(io.StringIO(uncompressed.decode("ascii")))}
    if stock_stages!=clock_stages:
        raise ValueError("TH10 clock and independent stock observer disagree on stages played")
    expected=describe(uncompressed,receipt["difficulty"],int(last["utc_ms"]))
    if any(evidence.get(k)!=v for k,v in expected.items()):
        raise ValueError("TH10 clock receipt disagrees with independent native rows")
    return expected["latest_observed_stage1_native_tick"]
