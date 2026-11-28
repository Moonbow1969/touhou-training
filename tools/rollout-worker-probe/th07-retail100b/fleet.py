"""Collect the frozen roster on two isolated stage lanes after both bootstrap checks."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import time

from run_case import run
from seal import seal

base = Path(__file__).resolve().parent.parent


def collect(stage):
    for source_id in (2, 3, 4, 6, 9):
        for attempt in range(2):
            suffix = '' if attempt == 0 else '-technical-retry'
            subdir = f'corpus/source{source_id:02d}-s{stage}{suffix}'
            output = base / subdir
            try:
                occupied = int(subprocess.check_output(['du', '-sb', str(base)]).split()[0])
                if occupied > 920_000_000 or shutil.disk_usage(base).free < 2_000_000_000:
                    raise RuntimeError('Disk budget exceeded')
                if not (output / 'capture.json').exists():
                    run(f'public-{source_id:02d}.rpy', stage, subdir)
                seal(base, output, source_id, None)
                break
            except Exception as exc:
                record = dict(source_id=source_id, stage=stage, attempt=attempt,
                              path=subdir, error=repr(exc), utc=time.time())
                with (base / f'technical-stage{stage}.jsonl').open('a') as f:
                    f.write(json.dumps(record) + '\n')
                if attempt == 1 or isinstance(exc, RuntimeError):
                    raise


if __name__ == '__main__':
    for stage in (1, 2):
        receipt = json.loads((base / f'preflight/green-v2-s{stage}/source-audit.json').read_text())
        assert receipt['admitted'] and receipt['stage'] == stage
        assert receipt['rows'] == receipt['native_current_equal'] == receipt['native_previous_equal'] == receipt['cursor_equal_expected']
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(collect, (1, 2)))
