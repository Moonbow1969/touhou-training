"""In-memory proof process for the private QEMU virtio serial channel."""

import json
import os
from pathlib import Path
import time
import uuid


port = Path("/dev/virtio-ports/org.moonbow.counter")
while not port.exists():
    time.sleep(.1)

state = {
    "nonce": uuid.uuid4().hex,
    "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    "pid": os.getpid(),
    "counter": 1729,
}

with port.open("r+b", buffering=0) as channel:
    while True:
        command = channel.readline().decode("ascii").strip().split()
        if not command:
            time.sleep(.05)
            continue
        if len(command) == 2 and command[0] == "add":
            state["counter"] += int(command[1])
        elif command != ["get"]:
            channel.write(b'{"error":"unknown command"}\n')
            continue
        channel.write(json.dumps(state, sort_keys=True).encode("ascii") + b"\n")
