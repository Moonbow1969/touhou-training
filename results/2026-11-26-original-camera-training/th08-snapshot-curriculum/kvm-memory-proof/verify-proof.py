"""Validate two full-VM rewinds of the same live guest process."""

import hashlib
import json
from pathlib import Path
import socket
import time


ROOT = Path("/var/tmp/moonbow-th08-qemu-261127")
SNAPSHOT = "counter-anchor-1746"


def connect(name, timeout):
    client = socket.socket(socket.AF_UNIX)
    client.settimeout(timeout)
    client.connect(str(ROOT / name))
    return client


def guest(command, timeout=3):
    with connect("counter.sock", timeout) as client, client.makefile("rb") as stream:
        client.sendall(command.encode("ascii") + b"\n")
        response = json.loads(stream.readline())
    if set(response) != {"nonce", "boot_id", "pid", "counter"}:
        raise ValueError(f"Unexpected counter reply: {response}")
    return response


def wait_guest():
    expires = time.monotonic() + 150
    while time.monotonic() < expires:
        try:
            return guest("get")
        except (OSError, ValueError):
            time.sleep(.15)
    raise TimeoutError("Guest cloud-init did not open the proof channel")


def qmp(client, stream, command, arguments=None):
    request = dict(execute=command)
    if arguments is not None:
        request["arguments"] = arguments
    client.sendall(json.dumps(request).encode("ascii") + b"\r\n")
    while True:
        response = json.loads(stream.readline())
        if "error" in response:
            raise RuntimeError(response["error"])
        if "return" in response:
            return response["return"]


def hmp(client, stream, command):
    result = qmp(client, stream, "human-monitor-command", {"command-line": command})
    if any(token in result.lower() for token in ("failed", "error", "does not support", "not found")):
        raise RuntimeError(result)
    return result


def changed(original, offset):
    return dict(original, counter=original["counter"] + offset)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def main():
    initial = wait_guest()
    require(initial["counter"] == 1729, "Proof guest must be a fresh process")
    boot_start = int((ROOT / "evidence/boot-start-real-ns").read_text())
    report = dict(qemu_guest_boot_to_first_query_s=(time.time_ns() - boot_start) / 1e9,
                  initial=initial, cpu_count=2, memory_mib=2048)
    with connect("qmp.sock", 90) as client, client.makefile("rb") as stream:
        report["qmp_banner"] = json.loads(stream.readline())
        qmp(client, stream, "qmp_capabilities")
        report["acceleration"] = qmp(client, stream, "query-kvm")
        require(report["acceleration"] == {"enabled": True, "present": True}, "QEMU did not use KVM")
        anchor = guest("add 17")
        require(anchor == changed(initial, 17), "Failed to establish live in-memory anchor")
        qmp(client, stream, "stop")
        require(not qmp(client, stream, "query-status")["running"], "VM did not stop at snapshot boundary")
        start = time.perf_counter_ns()
        hmp(client, stream, f"savevm {SNAPSHOT}")
        report["save_ms"] = (time.perf_counter_ns() - start) / 1e6
        report["snapshot_listing"] = hmp(client, stream, "info snapshots")
        require(SNAPSHOT in report["snapshot_listing"], "Snapshot tag missing after save")
        qmp(client, stream, "cont")
        report["saved_state"] = anchor
        report["saved_state_sha256"] = canonical_digest(anchor)
        report["restore_trials"] = []
        for trial, offset in enumerate((1009, 669), start=1):
            before = guest("add " + str(offset))
            require(before == changed(anchor if trial == 1 else changed(anchor, 77), offset),
                    "Guest state was not independently perturbed before restore")
            qmp(client, stream, "stop")
            start = time.perf_counter_ns()
            hmp(client, stream, f"loadvm {SNAPSHOT}")
            restore_ms = (time.perf_counter_ns() - start) / 1e6
            if not qmp(client, stream, "query-status")["running"]:
                qmp(client, stream, "cont")
            actual = guest("get")
            require(actual == anchor, "Restored process memory did not exactly match the snapshot")
            branch = guest("add 77")
            require(branch == changed(anchor, 77), "Restored branch did not reproduce the same counter delta")
            row = dict(trial=trial, before=before, restored=actual, restored_sha256=canonical_digest(actual),
                       repeated_branch=branch, repeated_branch_sha256=canonical_digest(branch), restore_ms=restore_ms)
            report["restore_trials"].append(row)
        qmp(client, stream, "quit")
    report["proof_passed"] = True
    out = ROOT / "evidence/proof.json"
    with out.open("x", encoding="utf-8") as result:
        result.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
