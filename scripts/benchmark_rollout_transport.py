"""Measure the rollout packet format on a real opening trace and dense stress data."""

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Thread
from time import perf_counter

import numpy as np
import zmq

from scripts.benchmark_trial_workers import TrialWorker, rates, require_stage
from scripts.rollout_transport import (
    FragmentKey, FragmentSpool, Receipt, SpoolFull, pack_fragment, rollout_socket, schema, unpack_fragment,
)


def capture_opening(command):
    columns = {key: np.zeros(shape, dtype=dtype) for key, (dtype, shape) in schema(256, 39).items()}
    counts = []
    with TrialWorker(command) as worker:
        state = worker.read(timeout=90)
        for index in range(256):
            if index == 128:
                state = worker.request("R")
            require_stage(state)
            side = (index // 20) % 2
            action = (5 if side else 3, 1, 1, 0)
            bullets = state["bullets"]
            counts.append(len(bullets))
            if bullets:
                columns["bullets"][index, :len(bullets), :6] = (
                    np.asarray(bullets, dtype=np.float32) / [384, 448, 16, 16, 32, 1]
                )
                columns["bullets"][index, :len(bullets), 6] = 1
            x, y = state["player"]
            columns["context"][index, :5] = x / 192, y / 448, state["lives"] / 8, len(bullets) / 96, state["frame"] / 2048
            columns["episode_start"][index] = index % 128 == 0
            columns["action"][index] = action
            state = worker.request("S 50" if side else "S 49")
            columns["native_delta"][index] = state["delta"]
    return columns, counts


def loopback(parts, receipt, repeats):
    context = zmq.Context()
    ready = Queue()
    failures = Queue()

    def learner():
        try:
            with rollout_socket(context, zmq.ROUTER) as server:
                server.setsockopt(zmq.LINGER, 1000)
                port = server.bind_to_random_port("tcp://127.0.0.1")
                ready.put(port)
                for _ in range(repeats):
                    identity, *request = server.recv_multipart()
                    _, _, result = unpack_fragment(request)
                    server.send_multipart([identity, result.pack()])
        except BaseException as error:
            failures.put(error)

    thread = Thread(target=learner, daemon=True)
    thread.start()
    seconds = []
    try:
        with rollout_socket(context, zmq.DEALER) as actor:
            actor.connect(f"tcp://127.0.0.1:{ready.get(timeout=5)}")
            for index in range(repeats):
                started = perf_counter()
                actor.send_multipart(list(parts))
                try:
                    ack = Receipt.unpack(actor.recv())
                except zmq.Again as error:
                    if not failures.empty():
                        raise failures.get() from error
                    raise RuntimeError(f"No loopback ACK for fragment {index + 1}.") from error
                if ack != receipt:
                    raise RuntimeError("Unexpected loopback ACK.")
                seconds.append(perf_counter() - started)
        thread.join(timeout=5)
        if not failures.empty():
            raise failures.get()
    finally:
        context.term()
    return rates(seconds)


def codec_benchmark(key, columns, *, compress, repeats):
    encode, decode = [], []
    for _ in range(repeats):
        start = perf_counter()
        parts, receipt = pack_fragment(key, columns, compress=compress)
        encode.append(perf_counter() - start)
        start = perf_counter()
        decoded_key, decoded, decoded_receipt = unpack_fragment(parts, policy=key.policy, environment=key.environment)
        decode.append(perf_counter() - start)
    assert decoded_key == key and decoded_receipt == receipt
    assert all(np.array_equal(column, decoded[name]) for name, column in columns.items())
    return {"bytes": sum(map(len, parts)), "encode": rates(encode), "decode": rates(decode),
            "loopback_validated_ack": loopback(parts, receipt, repeats)}


def expect_error(error_type, operation):
    try:
        operation()
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__}.")


def spool_benchmark(key, columns, parent):
    with TemporaryDirectory(prefix="rollout-spool-", dir=parent) as directory:
        spool = FragmentSpool(directory)
        first, first_ack = pack_fragment(key, columns)
        second, second_ack = pack_fragment(replace(key, sequence=1), columns)
        third, _ = pack_fragment(replace(key, sequence=2), columns)
        assert spool.put(first) == spool.put(first) == first_ack
        assert spool.put(second) == second_ack
        recovered = FragmentSpool(directory).pending()
        assert [row[2] for row in recovered] == [first_ack, second_ack]
        expect_error(SpoolFull, lambda: spool.put(third))
        expect_error(ValueError, lambda: spool.acknowledge(replace(first_ack, checksum="0" * 64)))
        expect_error(ValueError, lambda: unpack_fragment(first, policy="0" * 64))
        changed = dict(columns, reward=np.ones_like(columns["reward"]))
        conflicting, _ = pack_fragment(key, changed)
        expect_error(ValueError, lambda: spool.put(conflicting))
        assert spool.acknowledge(first_ack) and spool.acknowledge(second_ack)
        assert not spool.acknowledge(second_ack)
        seconds = []
        for _ in range(20):
            begin = perf_counter()
            receipt = spool.put(first)
            seconds.append(perf_counter() - begin)
            assert spool.acknowledge(receipt)
        if list(Path(directory).glob("*.packet")):
            raise AssertionError("Acknowledged fragments remained on disk.")
        expect_error(SpoolFull, lambda: FragmentSpool(directory, max_bytes=1).put(first))
        expect_error(SpoolFull, lambda: FragmentSpool(directory, reserve_bytes=2**60).put(first))
        return {"durable_put": rates(seconds), "samples": len(seconds), "sender_cap_test": "passed",
                "ack_replay_test": "passed", "identity_conflict_test": "passed", "stale_policy_test": "passed",
                "restart_recovery_test": "passed", "byte_cap_and_reserve_test": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=Path, required=True)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    commands = json.loads(args.workers.read_text(encoding="utf-8"))
    opening, counts = capture_opening(commands[args.worker])
    rng = np.random.default_rng(61)
    dense = dict(opening, bullets=rng.uniform(-1, 1, size=(256, 96, 7)).astype("<f4"))
    dense["bullets"][:, :, 6] = 1
    key = FragmentKey("wire-pilot", "benchmark", "fixed-opening",
                      sha256(b"no rollout policy").hexdigest(), sha256(b"TH14 0.01b transport pilot").hexdigest(), 0, 0)
    report = {
        "game": "TH14 0.01b", "decisions": 256, "context_width": 39,
        "opening_max_projectiles": max(counts), "opening_mean_projectiles": float(np.mean(counts)),
        "scope": "Real early-stage observations, zero placeholder value/reward columns, and synthetic dense stress data.",
        "software": {"pyzmq": zmq.__version__, "libzmq": zmq.zmq_version()},
        "cases": {
            label: {codec: codec_benchmark(key, columns, compress=(codec != "raw"), repeats=30)
                    for codec in ("raw", "zstd_level_1")}
            for label, columns in (("native_opening", opening), ("synthetic_dense", dense))
        },
        "spool": spool_benchmark(key, opening, args.out.parent),
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
