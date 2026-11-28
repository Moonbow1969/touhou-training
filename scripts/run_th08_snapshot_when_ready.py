"""Start the registered PPO fit when the mixed original manifest is sealed."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest","parent","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--timeout-seconds",type=float,default=7200)
    args = parser.parse_args()
    until = time.monotonic() + args.timeout_seconds
    while not args.manifest.is_file():
        if time.monotonic() >= until:
            raise TimeoutError("Mixed original collection did not produce its sealed manifest")
        print("Waiting for the registered ordinary and restored original cohorts",flush=True)
        time.sleep(30)
    payload = json.loads(args.manifest.read_bytes())
    print(f"Sealed mixed-input manifest found: {len(payload['rows'])} qualified trajectory rows",flush=True)
    subprocess.run([sys.executable,"-u","-m","scripts.train_th08_snapshot_ppo",
                    "--manifest",str(args.manifest),"--parent",str(args.parent),"--out",str(args.out)],check=True)


if __name__ == "__main__":
    main()
