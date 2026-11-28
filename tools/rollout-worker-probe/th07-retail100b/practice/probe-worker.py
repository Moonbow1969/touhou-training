"""Exercise the worker handoff with a read-only native observer and zero game input."""
import os
from pathlib import Path
import subprocess
import time

out=Path(os.environ["TH07_EVIDENCE"])
host=["wsl","-d",os.environ["TH07_WSL"],"--exec"] if os.environ["TH07_WSL"] else []
(out/"worker-ready").touch()
until=time.monotonic()+40
while not(out/"menu-released").is_file():
    if time.monotonic()>until:raise TimeoutError("menu did not release")
    time.sleep(.002)
subprocess.run([*host,"docker","exec",os.environ["TH07_PRACTICE_CONTAINER"],"sh","-ec",
                "/stageprobe >/capture/native.csv"],check=True)
