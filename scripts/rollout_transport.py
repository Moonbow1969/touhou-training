"""Bounded ZeroMQ fragment messages and a durable, scoped fragment spool."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
import os
from pathlib import Path
import re
import shutil
import tempfile

import msgpack
import numpy as np
import zmq
import zstandard


MAX_STEPS = 256
MAX_HEADER = 4096
MAX_BODY = 1024 * 1024
LABEL = re.compile(r"[A-Za-z0-9_.-]{1,64}", re.ASCII)
DIGEST = re.compile(r"[0-9a-f]{64}", re.ASCII)


@dataclass(frozen=True)
class FragmentKey:
    run: str
    worker: str
    lease: str
    policy: str
    environment: str
    sequence: int
    first_step: int

    def validate(self):
        for value in (self.run, self.worker, self.lease):
            if not isinstance(value, str) or not LABEL.fullmatch(value):
                raise ValueError("Invalid rollout label.")
        for value in (self.policy, self.environment):
            if not isinstance(value, str) or not DIGEST.fullmatch(value):
                raise ValueError("Invalid rollout fingerprint.")
        for value in (self.sequence, self.first_step):
            if type(value) is not int or not 0 <= value < 2**63:
                raise ValueError("Invalid rollout position.")

    def fragment_id(self):
        return sha256(msgpack.packb(tuple(asdict(self).values()), use_bin_type=True)).hexdigest()


@dataclass(frozen=True)
class Receipt:
    fragment_id: str
    checksum: str

    def validate(self):
        for field in (self.fragment_id, self.checksum):
            if not isinstance(field, str) or not DIGEST.fullmatch(field):
                raise ValueError("Invalid fragment receipt.")

    def pack(self):
        self.validate()
        return msgpack.packb({"type": "ack", **asdict(self)}, use_bin_type=True)

    @staticmethod
    def unpack(data):
        fields = decode_header(data)
        if fields.pop("type", None) != "ack":
            raise ValueError("Expected a fragment ACK.")
        receipt = Receipt(**fields)
        receipt.validate()
        return receipt


def schema(count, context_width):
    if type(count) is not int or not 1 <= count <= MAX_STEPS:
        raise ValueError("Invalid decision count.")
    if type(context_width) is not int or not 1 <= context_width <= 256:
        raise ValueError("Invalid context width.")
    return {
        "bullets": ("<f4", (count, 96, 7)), "context": ("<f4", (count, context_width)),
        "action": ("u1", (count, 4)), "log_prob": ("<f4", (count,)),
        "value": ("<f4", (count,)), "next_value": ("<f4", (count,)),
        "reward": ("<f4", (count,)), "terminated": ("u1", (count,)),
        "truncated": ("u1", (count,)), "episode_start": ("u1", (count,)),
        "native_delta": ("u1", (count,)),
    }


def decode_header(encoded):
    if not 0 < len(encoded) <= MAX_HEADER:
        raise ValueError("Invalid fragment header length.")
    header = msgpack.unpackb(encoded, raw=False, strict_map_key=True)
    if not isinstance(header, dict):
        raise ValueError("Invalid fragment header.")
    return header


def validate_columns(columns):
    for name in ("bullets", "context", "log_prob", "value", "next_value", "reward"):
        if not np.isfinite(columns[name]).all():
            raise ValueError(f"Non-finite values in {name}.")
    if (columns["action"] >= np.array([9, 2, 2, 2], dtype=np.uint8)).any():
        raise ValueError("Invalid action code.")
    for name in ("terminated", "truncated", "episode_start"):
        if (columns[name] > 1).any():
            raise ValueError(f"Invalid flag in {name}.")
    if (columns["native_delta"] == 0).any():
        raise ValueError("A rollout decision must advance a native tick.")


def pack_fragment(key, columns, *, compress=True):
    key.validate()
    count, width = columns["context"].shape
    layout = schema(count, width)
    if columns.keys() != layout.keys():
        raise ValueError("Unexpected fragment columns.")
    pieces = []
    for name, (dtype, shape) in layout.items():
        array = np.asarray(columns[name])
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise ValueError(f"Invalid shape or dtype for {name}.")
        pieces.append(array.tobytes(order="C"))
    validate_columns(columns)
    raw = b"".join(pieces)
    checksum = sha256(raw).hexdigest()
    compressed = zstandard.ZstdCompressor(level=1).compress(raw) if compress else raw
    codec = "zstd" if len(compressed) < len(raw) else "raw"
    header = msgpack.packb({
        "type": "rollout", "version": 1, "key": asdict(key), "count": count,
        "context_width": width, "codec": codec, "checksum": checksum,
    }, use_bin_type=True)
    return (header, compressed if codec == "zstd" else raw), Receipt(key.fragment_id(), checksum)


def unpack_fragment(parts, *, policy=None, environment=None):
    if len(parts) != 2:
        raise ValueError("Expected a header and a payload.")
    header_bytes, body = parts
    if not 0 < len(body) <= MAX_BODY:
        raise ValueError("Invalid fragment body length.")
    header = decode_header(header_bytes)
    if header["type"] != "rollout" or header["version"] != 1:
        raise ValueError("Unsupported fragment schema.")
    key = FragmentKey(**header["key"])
    key.validate()
    if (policy is not None and key.policy != policy) or (environment is not None and key.environment != environment):
        raise ValueError("Fragment does not match the current policy and environment.")
    layout = schema(header["count"], header["context_width"])
    sizes = {name: np.dtype(dtype).itemsize * int(np.prod(shape)) for name, (dtype, shape) in layout.items()}
    expected = sum(sizes.values())
    if header["codec"] == "zstd":
        frame = zstandard.get_frame_parameters(body)
        if frame.content_size != expected or frame.window_size > MAX_BODY:
            raise ValueError("Invalid compressed fragment size.")
        raw = zstandard.ZstdDecompressor().decompress(body, max_output_size=expected, allow_extra_data=False)
    elif header["codec"] == "raw":
        raw = body
    else:
        raise ValueError("Unsupported fragment compression.")
    checksum = sha256(raw).hexdigest()
    if len(raw) != expected or checksum != header["checksum"]:
        raise ValueError("Fragment checksum or decoded size mismatch.")
    columns = {}
    offset = 0
    for name, (dtype, shape) in layout.items():
        columns[name] = np.frombuffer(raw, dtype=dtype, count=sizes[name] // np.dtype(dtype).itemsize,
                                      offset=offset).reshape(shape)
        offset += sizes[name]
    validate_columns(columns)
    return key, columns, Receipt(key.fragment_id(), checksum)


def rollout_socket(context, role, *, high_water_mark=2, timeout_ms=2000):
    if role not in (zmq.DEALER, zmq.ROUTER):
        raise ValueError("Use a DEALER actor or ROUTER learner.")
    socket = context.socket(role)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.MAXMSGSIZE, MAX_BODY)
    socket.setsockopt(zmq.SNDHWM, high_water_mark)
    socket.setsockopt(zmq.RCVHWM, high_water_mark)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.HEARTBEAT_IVL, 1000)
    socket.setsockopt(zmq.HEARTBEAT_TIMEOUT, 5000)
    socket.setsockopt(zmq.IMMEDIATE if role == zmq.DEALER else zmq.ROUTER_MANDATORY, 1)
    return socket


class SpoolFull(RuntimeError):
    """The actor should stop collection until a fragment is acknowledged."""


class FragmentSpool:
    def __init__(self, directory, *, max_fragments=2, max_bytes=16 * 1024**2, reserve_bytes=2 * 1024**3):
        if max_fragments < 1 or max_bytes < 1 or reserve_bytes < 0:
            raise ValueError("Invalid spool budget.")
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=True)
        self.max_fragments = max_fragments
        self.max_bytes = max_bytes
        self.reserve_bytes = reserve_bytes

    @contextmanager
    def _exclusive(self):
        with (self.path / ".spool-lock").open("a+b") as handle:
            if os.name == "nt":
                import msvcrt
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield

    def _read_parts(self, path):
        if path.stat().st_size > MAX_BODY + MAX_HEADER + 32:
            raise ValueError("Stored fragment exceeds its size bound.")
        return tuple(msgpack.unpackb(path.read_bytes(), raw=False))

    def _packets(self):
        return [path for path in self.path.glob("*.packet") if DIGEST.fullmatch(path.stem)]

    def pending(self):
        with self._exclusive():
            recovered = []
            for path in self._packets():
                parts = self._read_parts(path)
                key, _, receipt = unpack_fragment(parts)
                if receipt.fragment_id != path.stem:
                    raise ValueError("Stored fragment ID does not match its filename.")
                recovered.append((key, parts, receipt))
            return sorted(recovered, key=lambda item: item[0].sequence)

    def put(self, parts):
        _, _, receipt = unpack_fragment(parts)
        target = self.path / f"{receipt.fragment_id}.packet"
        with self._exclusive():
            if target.exists():
                if unpack_fragment(self._read_parts(target))[2] != receipt:
                    raise ValueError("Receipt conflict for an existing fragment ID.")
                return receipt
            # These files are the spool's own incomplete writes and there is no active writer under this lock.
            for orphan in self.path.glob(".fragment-*.tmp"):
                orphan.unlink()
            pending = self._packets()
            encoded = msgpack.packb(list(parts), use_bin_type=True)
            if (len(pending) >= self.max_fragments
                    or sum(path.stat().st_size for path in pending) + len(encoded) > self.max_bytes
                    or shutil.disk_usage(self.path).free - len(encoded) < self.reserve_bytes):
                raise SpoolFull("Pending-rollout budget or free-space reserve exhausted.")
            with tempfile.NamedTemporaryFile(dir=self.path, prefix=".fragment-", suffix=".tmp", delete=False) as tmp:
                temporary = Path(tmp.name)
                try:
                    tmp.write(encoded)
                    tmp.flush()
                    os.fsync(tmp.fileno())
                except BaseException:
                    tmp.close()
                    temporary.unlink(missing_ok=True)
                    raise
            try:
                temporary.replace(target)
                if os.name != "nt":
                    fd = os.open(self.path, os.O_RDONLY)
                    try:
                        os.fsync(fd)
                    finally:
                        os.close(fd)
            finally:
                temporary.unlink(missing_ok=True)
        return receipt

    def acknowledge(self, receipt):
        receipt.validate()
        target = self.path / f"{receipt.fragment_id}.packet"
        with self._exclusive():
            if not target.exists():
                return False
            if unpack_fragment(self._read_parts(target))[2] != receipt:
                raise ValueError("Refusing to remove a fragment for a different receipt.")
            target.unlink()
            return True
