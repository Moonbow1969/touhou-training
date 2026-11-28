"""Offline reference for a bounded single-producer TH08 native archive.

The producer offers owned native bytes. Only the worker or the finalizer opens
files. Filenames exposed to the existing scorer are published on explicit seal.
The caller must wait for the final receipt before stopping the container.
"""
import hashlib
import json
import os
from pathlib import Path
import struct
import threading
import zlib

MAX_WORDS = 65534
HEADER = struct.Struct("<II")  # retained old words, new total words
U32 = struct.Struct("<I")


def due_count(stage_index, tick, mode, flags, stable, origin, cursor, previous):
    """Mirror reader.c's eligibility after its consistent original-state read."""
    count = tick + int(flags == 0)
    if (stable and mode == 0 and 0 <= stage_index < 3 and 0 < count <= MAX_WORDS
            and origin and cursor == origin + 36 + 2 * tick
            and ((flags == 4 and count >= previous + 32) or (flags == 0 and count > previous))):
        return count
    return None


def recover(journal):
    """Return the last checksum-complete prefix. An interrupted tail is unsealed."""
    current = b""
    incomplete = False
    with Path(journal).open("rb") as f:
        while head := f.read(HEADER.size):
            if len(head) < HEADER.size:
                incomplete = True
                break
            kept, count = HEADER.unpack(head)
            if not (kept <= len(current)//2 < count <= MAX_WORDS):
                raise ValueError("Invalid native journal extent")
            suffix = f.read(2 * (count - kept))
            check = f.read(U32.size)
            if len(suffix) < 2*(count-kept) or len(check) < U32.size:
                incomplete = True
                break
            if U32.unpack(check)[0] != zlib.crc32(suffix, zlib.crc32(head)):
                raise ValueError("Corrupt native journal record")
            current = current[:2*kept] + suffix
    return current, incomplete


class NativeArchive:
    """One isolated output directory per credit. Never reuse a prior spool."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(exist_ok=False)
        self.cv = threading.Condition()
        self.pending = {}  # one immutable latest prefix per stage
        self.extents = {}
        self.closed = False
        self.error = None
        self.coalesced = 0
        self.high_water = 0
        self.records = 0
        self.worker = threading.Thread(target=self._run, name="native-archive")
        self.worker.start()

    def offer(self, stage, source):
        """Single producer. source is the exact validated little-endian word slice."""
        n = len(source)
        if stage not in (1, 2, 3) or n % 2 or not 0 < n//2 <= MAX_WORDS:
            raise ValueError("Invalid native capture")
        owned = bytes(source)
        with self.cv:
            if self.error is not None:
                raise RuntimeError("Archival worker failed") from self.error
            if self.closed or n//2 <= self.extents.get(stage, 0):
                raise ValueError("Closed archive or non-increasing native capture")
            if stage in self.pending:
                self.coalesced += 1
            self.pending[stage] = owned
            self.extents[stage] = n//2
            self.high_water = max(self.high_water, sum(map(len,self.pending.values())))
            self.cv.notify()

    def _append(self, stage, old, new):
        # Usually only the newly recorded words need writing. If the original
        # revises an earlier word, journal the entire replacement prefix.
        kept = len(old)//2 if new.startswith(old) else 0
        suffix = memoryview(new)[2*kept:]
        path = self.path/f"stage-{stage}.journal"
        # Bound disk if a changing early native word forces frequent replacements.
        compact = path.exists() and path.stat().st_size + len(suffix) + 12 > 4*MAX_WORDS + 4096
        if compact:
            kept = 0
            suffix = memoryview(new)
        head = HEADER.pack(kept, len(new)//2)
        crc = zlib.crc32(suffix, zlib.crc32(head))
        target = path.with_suffix(".new") if compact else path
        with target.open("wb" if compact else "ab") as f:
            f.write(head)
            f.write(suffix)
            f.write(U32.pack(crc))
        if compact:
            os.replace(target, path)
        self.records += 1

    def _run(self):
        previous = {}
        try:
            while True:
                with self.cv:
                    self.cv.wait_for(lambda: self.closed or bool(self.pending))
                    if not self.pending:
                        return
                    stage = next(iter(self.pending))
                    data = self.pending.pop(stage)
                self._append(stage, previous.get(stage,b""), data)
                previous[stage] = data
        except Exception as e:
            with self.cv:
                self.error = e
                self.cv.notify_all()

    def seal(self, expected_counts):
        """Call outside the hot loop with independently confirmed per-stage extents.

        A failed seal can leave diagnostic journals or individually published
        stage files. Only sealed.json accepts the per-credit set as complete.
        """
        with self.cv:
            if self.closed:
                raise RuntimeError("Archive already closed")
            self.closed = True
            self.cv.notify_all()
        self.worker.join()
        if self.error is not None:
            raise RuntimeError("Archival worker failed") from self.error
        if not expected_counts or expected_counts != self.extents:
            raise ValueError("Terminal or stage evidence does not match submitted extents")
        manifest = {}
        for stage,count in sorted(expected_counts.items()):
            words, partial = recover(self.path/f"stage-{stage}.journal")
            if partial or len(words) != count*2:
                raise ValueError("Uncommitted or short final journal")
            filename = f"native-record-stage-{stage}.bin"
            data = U32.pack(count) + words
            self._publish(filename, data)
            manifest[filename] = {"words":count, "sha256":hashlib.sha256(data).hexdigest()}
        receipt = {"files":manifest, "coalesced":self.coalesced,"journal_records":self.records,
                   "peak_pending_bytes":self.high_water}
        self._publish("sealed.json", json.dumps(receipt,sort_keys=True).encode("ascii"))
        return receipt

    def _publish(self, name, data):
        tmp = self.path/(name+".partial")
        with tmp.open("xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path/name)
