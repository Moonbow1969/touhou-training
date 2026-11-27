"""Make an independent TH08 Linux entry point that starts the existing visual title check immediately."""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(entry):
    wait="sleep 10\n"
    gate="test -s /capture/title-gate.json"
    accepted="cp /capture/title-accepted.png /capture/title.png"
    first_menu_key="tap z; sleep 1; snap default-rank"
    if entry.count(wait)!=1 or entry.count(gate)!=1 or entry.count(accepted)!=1 or entry.count(first_menu_key)!=1:
        raise ValueError("Expected the validated TH08 selected-Start observer and rank transition")
    if not entry.index(wait)<entry.index(gate)<entry.index(accepted)<entry.index(first_menu_key):
        raise ValueError("The actual title certificate must gate the first menu keystroke")
    return entry.replace(wait,"date -Ins > /capture/window-ready.txt\nsleep 0\n")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("verified_entry",type=Path)
    p.add_argument("new_entry",type=Path)
    a=p.parse_args()
    if a.verified_entry.resolve()==a.new_entry.resolve():p.error("Provide a separate output path")
    data=a.verified_entry.read_bytes()
    result=prepare(data.decode("utf8")).encode("utf8")
    if a.new_entry.exists() and a.new_entry.read_bytes()!=result:p.error("Another entry already exists at the output path")
    a.new_entry.write_bytes(result);a.new_entry.chmod(a.verified_entry.stat().st_mode)
    print(json.dumps(dict(parent_sha256=hashlib.sha256(data).hexdigest(),new_sha256=hashlib.sha256(result).hexdigest())))


if __name__=="__main__":main()
