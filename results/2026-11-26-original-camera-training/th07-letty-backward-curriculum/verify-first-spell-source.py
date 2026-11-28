"""Verify this prospective three-phase practice start against the retail TH07 stage script.

Pass the directory containing ecldata1.ecl and its exact thtk 12 thecl -d7 dump.
The original game and script are not packaged with this study.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct


def check(source_dir):
    raw = (source_dir / "ecldata1.ecl").read_bytes()
    dump = (source_dir / "native-ecl.dump").read_bytes()
    for data, expected in [
        (raw, "c443e28b86cb20eeb7613d786c7ff667c2bd39726bf7ffbb38a3c0e354db1b40"),
        (dump, "8d8f469be67f7e305fa309735fb0c9f4f01512ed248462b0e559d8ba4228eb35"),
    ]:
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Unrecognized retail stage script or decoder output")
    text = dump.decode("cp932")
    subs = dict(re.findall(r"sub Sub(\d+)\(\)\s*\{(.*?)\n\}", text, re.S))
    if struct.unpack_from("<IHH", raw, 0x4c20) != (60, 41, 16):
        raise ValueError("Unexpected original BOSS2 warp instruction")
    expected_ops = {
        "38": ["ins_148(0, 1700, 42)", "ins_115(42)", "ins_107(39)"],
        "42": ["ins_30([10016], 1, 240, Sub42_696)", "ins_114(3000)",
               "ins_133()", 'ins_90(0, 3, "寒符「リンガリングコールド」")'],
        "39": ["ins_91()", "ins_112(2000)", "!EN    ins_113(48)",
               "ins_114(2400)", "!EN    ins_115(48)", "ins_107(51)"],
        "48": ["ins_30([10016], 1, 240, Sub48_676)", "ins_114(3000)",
               "ins_133()", 'ins_90(0, 7, "冬符「フラワーウィザラウェイ」")'],
        "51": ["ins_91()"],
    }
    for sub, ordered in expected_ops.items():
        remaining = subs[sub]
        for op in ordered:
            start = remaining.find(op)
            if start < 0:
                raise ValueError(f"Missing or reordered {op} in Sub{sub}")
            remaining = remaining[start + len(op):]
    cap = 12_500
    combat_and_transitions = 60 + 3000 + 2400 + 3000 + 60
    envelope = combat_and_transitions + 1680
    loose_envelope = combat_and_transitions + 2580
    return dict(phase_subs=[42, 39, 48], phase_active_ids=[3, None, 7],
                envelope=envelope, loose_envelope=loose_envelope,
                margin_over_loose_pct=round(100*(cap/loose_envelope-1), 2),
                elapsed_native_tick_cap=cap, decision_cap=cap//4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    print(json.dumps(check(parser.parse_args().source_dir), indent=2))
