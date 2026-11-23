"""Run a training or evaluation command and preserve its provenance and output."""

import argparse
from datetime import datetime, timezone
from hashlib import file_digest
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tarfile
from time import perf_counter, sleep
from uuid import uuid4


PROJECT = Path(__file__).resolve().parents[1]
PACKAGES = ('numpy', 'torch', 'gymnasium', 'stable-baselines3')


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def command_output(command):
    try:
        result = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def repository(run):
    commit = command_output(['git', 'rev-parse', 'HEAD'])
    status = command_output(['git', 'status', '--porcelain=v1', '--untracked-files=all'])
    if commit and status:
        paths = command_output(['git', '-c', 'core.quotePath=false', 'ls-files', '--cached', '--others',
                                '--exclude-standard', '--', '*.py', 'requirements.txt'])
        with tarfile.open(run / 'source.tar.gz', 'w:gz') as archive:
            for name in sorted(set((paths or '').splitlines())):
                path = PROJECT / name
                if path.is_file() and not path.is_symlink():
                    archive.add(path, arcname=name)
    return {'commit': commit, 'dirty': None if status is None else bool(status),
            'status': status.splitlines() if status else []}


def environment():
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    gpu = command_output(['nvidia-smi', '--query-gpu=index,name,driver_version,memory.total', '--format=csv,noheader'])
    return {'os': platform.platform(), 'python': platform.python_version(), 'packages': packages,
            'gpus': gpu.splitlines() if gpu else [],
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}


def fingerprint(path):
    with path.open('rb') as stream:
        digest = file_digest(stream, 'sha256').hexdigest()
    return {'bytes': path.stat().st_size, 'sha256': digest}


def inventory(roots):
    rows = []
    for original_path in roots:
        root = original_path.resolve()
        files = sorted(path for path in root.rglob('*') if path.is_file()) if root.is_dir() else [root]
        entries = []
        for path in files:
            try:
                info = fingerprint(path)
            except OSError as exc:
                info = {'error': str(exc)}
            entries.append({'path': path.relative_to(root).as_posix() if path != root else root.name, **info})
        rows.append({'root': str(root), 'files': entries})
    return rows


def write_json(path, content):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(content, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if os.name != 'nt' or attempt == 4:
                raise
            sleep(.05)


def execute(command, log):
    child_env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=child_env) as process:
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line.decode('utf-8', errors='replace'), end='', flush=True)
            return process.wait()
        except KeyboardInterrupt:
            if process.poll() is None:
                process.terminate()
            raise
        except BaseException:
            if process.poll() is None:
                process.terminate()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--kind', required=True, choices=('capture', 'sft', 'world', 'rl', 'evaluation', 'other'))
    parser.add_argument('--runs', type=Path, default=PROJECT / 'runs')
    parser.add_argument('--input', type=Path, action='append', default=[])
    parser.add_argument('--artifact', type=Path, action='append', default=[])
    parser.add_argument('--note', default='')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or not re.fullmatch('[a-z0-9][a-z0-9-]{0,79}', args.name):
        parser.error('Supply a lowercase alphanumeric run name with optional dashes and a command after --.')
    if any(not path.exists() for path in args.input):
        parser.error('Every --input path must exist before the command starts.')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = (args.runs / f'{stamp}-{args.name}-{uuid4().hex[:8]}').resolve()
    run.mkdir(parents=True)
    manifest = run / 'run.json'
    report = {'schema_version': 1, 'run': run.name, 'kind': args.kind, 'note': args.note, 'status': 'starting',
              'started_at_utc': utc_now(), 'cwd': str(Path.cwd()), 'command': command,
              'requested_artifacts': [str(path.resolve()) for path in args.artifact]}
    write_json(manifest, report)
    print(f'Run record {run}', flush=True)
    start = perf_counter()
    exit_code = 1
    try:
        report.update(source=repository(run), environment=environment(), inputs=inventory(args.input))
        report['status'] = 'running'
        write_json(manifest, report)
        with (run / 'console.log').open('wb') as log:
            exit_code = execute(command, log)
        report['status'] = 'completed' if exit_code == 0 else 'failed'
    except KeyboardInterrupt:
        exit_code = 130
        report['status'] = 'interrupted'
    except OSError as exc:
        report.update(status='failed', launch_error=str(exc))
        print(exc, file=sys.stderr)
    finally:
        report.update(finished_at_utc=utc_now(), elapsed_seconds=round(perf_counter() - start, 3),
                      exit_code=exit_code, artifacts=inventory(args.artifact))
        write_json(manifest, report)
    return exit_code if exit_code >= 0 else 128 - exit_code


if __name__ == '__main__':
    sys.exit(main())
