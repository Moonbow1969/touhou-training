"""Release only signed original observations, from one source stage at a time."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import zipfile

work = Path(__file__).resolve().parent
study = None
remote_base = '/var/tmp/moonbow-th07-retail-100b-preflight'
tag = 'camera-training-2026-11-26'
repo = 'Moonbow1969/touhou-training'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def publish(rel, source, stage):
    folder = study / 'stages' / f'{source:02d}-s{stage}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'release.json').is_file():
        return
    remote = remote_base + '/' + rel
    for name in ('source-audit.json', 'capture.json', 'sensor.log', 'build.sha256'):
        subprocess.run(['scp', '-q', f'momentum:{remote}/{name}', str(folder / name)], check=True)
    audit = json.loads((folder / 'source-audit.json').read_bytes())
    role = next(c for c in json.loads((study / 'protocol.json').read_bytes())['roles'] if c['id'] == source)
    assert audit['stage'] == stage and audit['original_replay_sha256'] == role['original_sha256']
    assert audit['role'] == role['role'] and audit['protocol_sha256'] == sha(study / 'protocol.json')
    members = dict(audit['artifacts_sha256'])
    for name in ('source-audit.json', 'capture.json', 'sensor.log'):
        members[name] = sha(folder / name)
    assert all(re.fullmatch(r'[a-zA-Z0-9.-]+', n) for n in members)
    assert not {'th07.exe', 'th07.dat', 'th7_01.rpy'}.intersection(members)
    archive_name = f'th07-retail-100b-human-{source:02d}-s{stage}-camera-input-xy.zip'
    with tempfile.TemporaryDirectory() as scratch:
        archive = Path(scratch) / archive_name
        read = ['ssh', 'momentum', 'tar -C ' + remote + ' -cf - ' + ' '.join(sorted(members))]
        with subprocess.Popen(read, stdout=subprocess.PIPE) as proc, zipfile.ZipFile(archive, 'x') as z:
            seen = set()
            with tarfile.open(fileobj=proc.stdout, mode='r|') as tar:
                for entry in tar:
                    n = entry.name
                    assert n in members and n not in seen and entry.isfile()
                    seen.add(n)
                    # Freeze ZIP timestamps so resuming an upload reproduces the exact bytes.
                    info = zipfile.ZipInfo(n)
                    with tar.extractfile(entry) as stream, z.open(info, 'w') as sink:
                        h = hashlib.sha256()
                        while block := stream.read(1024 * 1024):
                            h.update(block)
                            sink.write(block)
                    assert h.hexdigest() == members[n]
            assert proc.wait() == 0 and seen == set(members)
        with zipfile.ZipFile(archive) as z:
            for n in ('positions.npz', 'opening.png', 'selected-stage.png'):
                with z.open(n) as src, (folder / n).open('wb') as dst:
                    shutil.copyfileobj(src, dst)
        digest = sha(archive)
        query = ['gh', 'release', 'view', tag, '--repo', repo, '--json', 'assets',
                 '--jq', '.assets[] | select(.name=="' + archive_name + '")']
        answer = subprocess.check_output(query).strip()
        if not answer:
            subprocess.run(['gh', 'release', 'upload', tag, '--repo', repo, str(archive)], check=True)
            answer = subprocess.check_output(query).strip()
        asset = json.loads(answer)
        assert asset['digest'] == 'sha256:' + digest and asset['size'] == archive.stat().st_size
        receipt = dict(source_id=source, stage=stage, role=role['role'], author=role['author'],
                       native_path=rel, url=asset['url'], name=archive_name, sha256=digest,
                       bytes=asset['size'], source_audit_sha256=members['source-audit.json'])
        (folder / 'release.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(archive_name, asset['size'], digest, flush=True)


def main():
    protocol = json.loads((study / 'protocol.json').read_bytes())
    pending = {(q['id'], s) for q in protocol['roles'] for s in (1, 2)}
    while pending:
        cmd = f'cd {remote_base} && find preflight corpus -name source-audit.json'
        files = set(subprocess.check_output(['ssh', 'momentum', cmd], text=True).splitlines())
        for source, stage in sorted(pending):
            rel = f'preflight/green-v2-s{stage}' if source == 1 else f'corpus/source{source:02d}-s{stage}'
            options = [p for p in (rel, rel+'-technical-retry') if p+'/source-audit.json' in files]
            if options:
                assert len(options) == 1
                publish(options[0], source, stage)
                pending.remove((source, stage))
        if pending:
            time.sleep(25)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path, help='Local study root containing the registered protocol.json')
    study = parser.parse_args().study.resolve()
    main()
