"""Publish sealed human-camera training shards and verify GitHub's asset checksums."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import zipfile


def checksum(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'manifest', 'out', 'scratch'):
        p.add_argument('--' + key, required=True, type=Path)
    p.add_argument('--prior', type=Path)
    p.add_argument('--repository', required=True)
    p.add_argument('--release', required=True)
    args = p.parse_args()
    root = args.root.resolve()
    manifest = json.loads(args.manifest.read_bytes())
    rows = manifest['records']
    if manifest['split'] != 'train' or manifest['unique_new_pilots'] != len({r['player_group'] for r in rows}) or len(rows) != manifest['unique_new_pilots']:
        raise ValueError('Training manifest is incomplete')
    prior = json.loads(args.prior.read_bytes())['records'] if args.prior else []
    if prior != rows[:len(prior)] or len(prior) >= len(rows):
        raise ValueError('Prior release is not a strict prefix')
    paths = []
    sidecars = {}
    for row in rows[len(prior):]:
        path = (root / row['path']).resolve()
        path.relative_to(root)
        if row['split'] != 'train' or path.suffix != '.npz' or checksum(path) != row['archive_sha256']:
            raise ValueError('Changed human camera or non-training shard')
        sidecar=path.with_suffix('.json')
        metadata=json.loads(sidecar.read_bytes())
        if metadata.get('archive_sha256')!=row['archive_sha256'] or metadata['replay_sha256']!=row['replay_sha256']:
            raise ValueError('Camera metadata identifies another replay')
        relative=sidecar.relative_to(root)
        target=args.out.parent/'camera-metadata'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(sidecar,target)
        sidecars[relative.as_posix()]=checksum(target)
        paths.append((path, row['path']))
    def assets():
        raw = subprocess.check_output(['gh','release','view',args.release,'--repo',args.repository,'--json','assets'])
        return {a['name']: a for a in json.loads(raw)['assets']}
    args.scratch.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.scratch) as tmp:
        name = f'human-train-cameras-{len(prior)+1:03d}-{len(rows):03d}.zip'
        zpath = Path(tmp) / name
        with zipfile.ZipFile(zpath,'w',compression=zipfile.ZIP_STORED) as z:
            for path, member in [(args.manifest, args.manifest.name), *paths]:
                info = zipfile.ZipInfo(member)
                with path.open('rb') as src, z.open(info,'w') as dst:
                    shutil.copyfileobj(src,dst,1024*1024)
        digest = checksum(zpath)
        asset = assets().get(name)
        if asset is None:
            subprocess.run(['gh','release','upload',args.release,str(zpath),'--repo',args.repository],check=True)
            asset = assets()[name]
        if asset['digest'] != 'sha256:'+digest or asset['size'] != zpath.stat().st_size:
            raise ValueError('Remote bytes disagree with frozen training files')
    receipt = dict(asset=asset['url'], sha256=digest, bytes=asset['size'], unique_shards=len(paths),
                   new_action_windows=sum(r['sft_four_tick_decisions'] for r in rows[len(prior):]),
                   manifest_sha256=checksum(args.manifest), camera_metadata_sha256=sidecars,
                   preceding_manifest_sha256=checksum(args.prior) if prior else None)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(json.dumps(receipt),flush=True)


if __name__ == '__main__':
    main()
