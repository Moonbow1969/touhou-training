"""No-game byte/latency/failure fixtures for the proposed TH08 archive."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import threading
import time
import unittest

from read_archive_async import MAX_WORDS, NativeArchive, due_count, recover


class StallWriter(NativeArchive):
    def __init__(self,path):
        self.started = threading.Event()
        self.release = threading.Event()
        super().__init__(path)

    def _append(self, stage, old, new):
        self.started.set()
        if not self.release.wait(15):
            raise TimeoutError("Test did not release worker")
        super()._append(stage,old,new)


class ArchiveTests(unittest.TestCase):
    def test_requests_under_stalled_disk_and_exact_stage_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"spool"
            a=StallWriter(p)
            a.offer(1,bytes(64))
            self.assertTrue(a.started.wait(5))
            expected={}
            elapsed=time.perf_counter()
            try:
                for count in (64,96,128,130):
                    for stage in (1,2,3):
                        # Prove that native scratch may be overwritten immediately.
                        b=bytearray(bytes([stage,count%256])*count)
                        a.offer(stage,b)
                        expected[stage]=struct.pack("<I",count)+bytes(b)
                        b[:]=bytes(len(b))
                elapsed=time.perf_counter()-elapsed
                self.assertFalse(a.release.is_set())
                self.assertFalse((p/"sealed.json").exists())
            finally:
                a.release.set()
            receipt=a.seal({i:130 for i in (1,2,3)})
            self.assertLessEqual(receipt["peak_pending_bytes"],3*130*2)
            self.assertGreaterEqual(receipt["coalesced"],9)
            for stage,data in expected.items():
                self.assertEqual((p/f"native-record-stage-{stage}.bin").read_bytes(),data)
            print(f"12 offers returned during blocked disk in {1000*elapsed:.3f} ms")

    def test_torn_or_corrupt_journal_and_compaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"spool"; a=NativeArchive(p)
            # Force separate durable records, including a changed historical word.
            # Call the writer primitive without enqueueing to avoid coalescing.
            try:
                old=b""
                for j in range(4):
                    new=bytes([j+1,0])*(60000+j)
                    a._append(1,old,new);old=new
                journal=p/"stage-1.journal"
                self.assertLessEqual(journal.stat().st_size,4*MAX_WORDS+4096)
                self.assertEqual(recover(journal),(old,False))
                valid=journal.read_bytes()
                journal.write_bytes(valid+b"abc")
                self.assertEqual(recover(journal),(old,True))
                tail=struct.pack("<II",len(old)//2,len(old)//2+4)+b"data"
                journal.write_bytes(valid+tail)
                self.assertEqual(recover(journal),(old,True))
                bad=bytearray(valid);bad[-1]^=128
                journal.write_bytes(bad)
                with self.assertRaises(ValueError):recover(journal)
            finally:
                with self.assertRaises(ValueError):a.seal({})
            self.assertFalse((p/"sealed.json").exists())

    def test_write_error_and_short_final_are_unsealed(self):
        class Broken(NativeArchive):
            def _append(self,*args):raise OSError("simulated ENOSPC")
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"full";a=Broken(p);a.offer(1,bytes(64))
            with self.assertRaises(RuntimeError):a.seal({1:32})
            self.assertFalse((p/"sealed.json").exists())
            p=Path(tmp)/"short";a=NativeArchive(p);a.offer(1,bytes(64))
            with self.assertRaises(ValueError):a.seal({1:33})
            self.assertFalse((p/"sealed.json").exists())

    def test_native_reader_count_and_gate(self):
        o=0x1000
        def eligible(tick,flags,previous,**kw):
            params=dict(stage_index=0,tick=tick,mode=0,flags=flags,stable=True,origin=o,
                        cursor=o+36+2*tick,previous=previous)
            return due_count(**(params|kw))
        self.assertEqual(eligible(32,4,0),32)
        self.assertIsNone(eligible(33,4,32))
        self.assertEqual(eligible(3374,0,3360),3375)
        self.assertEqual(eligible(3375,0,3375),3376)
        self.assertIsNone(eligible(3375,0,3375,cursor=o+36+2*3375+2))
        self.assertIsNone(eligible(65534,0,0))
        self.assertIsNone(eligible(32,4,0,mode=1))
        self.assertIsNone(eligible(32,4,0,stable=False))


def replay(raw,tmp):
    expected={}
    for p in raw.glob("native-record-stage-*.bin"):
        stage=int(p.stem.split("-")[-1]);data=p.read_bytes();count=struct.unpack_from("<I",data)[0]
        assert len(data)==4+2*count
        expected[stage]=data
    assert expected
    a=NativeArchive(tmp);counts={};calls=0;legacy_written_bytes=0
    try:
        with (raw/"clock.tsv").open() as f:
            for r in csv.DictReader(f,delimiter=" "):
                stage=int(r["stage_index"])+1
                count=due_count(stage-1,int(r["tick"]),int(r["mode"]),int(r["flags"]),int(r["stable"]),
                                int(r["origin"]),int(r["cursor"]),counts.get(stage,0))
                if count is None:continue
                native=expected[stage][4:4+2*count]
                assert len(native)==2*count
                a.offer(stage,native);counts[stage]=count;calls+=1
                legacy_written_bytes+=len(native)+4
    finally:
        sealed=a.seal({s:(len(b)-4)//2 for s,b in expected.items()})
    for stage,b in expected.items():
        assert (tmp/f"native-record-stage-{stage}.bin").read_bytes()==b
    return dict(fixture=raw.parent.name,clock_sha256=hashlib.sha256((raw/"clock.tsv").read_bytes()).hexdigest(),
                snapshots_due=calls,legacy_snapshot_bytes=legacy_written_bytes,files=sealed["files"],
                journal_bytes=sum(p.stat().st_size for p in tmp.glob("stage-*.journal")),
                immutable_file_bytes=sum(len(v) for v in expected.values()),worker_records=sealed["journal_records"],
                coalesced=sealed["coalesced"],peak_pending_bytes=sealed["peak_pending_bytes"])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture",action="append",type=Path,default=[])
    parser.add_argument("--report",type=Path)
    args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ArchiveTests))
    if not result.wasSuccessful():raise SystemExit(1)
    with tempfile.TemporaryDirectory() as td:
        actual=[replay(f,Path(td)/str(i)) for i,f in enumerate(args.fixture)]
    report={"tests_passed":result.testsRun,"engineering_originals":actual,"deployment":"offline fixture only"}
    if args.report:args.report.write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report,indent=2))
