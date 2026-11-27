"""Fit a title-count by sampling-budget imitation comparison on the same GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def checksum(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True,type=Path)
    args=p.parse_args()
    root=args.bundle.resolve()
    plan=json.loads((root/'protocol.json').read_bytes())
    paths={}
    for row in plan['files']:
        source=Path(row['remote'])
        file=root/'named'/row['name']
        file.parent.mkdir(exist_ok=True)
        for raw,link,key in [(source,file,'sha256'),(source.with_suffix('.json'),file.with_suffix('.json'),'metadata_sha256')]:
            if checksum(raw)!=row[key]:raise ValueError(f'Changed source {raw}')
            if not link.exists():link.symlink_to(raw)
            if link.resolve()!=raw:raise ValueError(f'Conflicting source {link}')
        paths[row['sha256']]=file
    for key,source in [('encoder',Path(plan['encoder'])),('reference',root/'reference.bin')]:
        if checksum(source)!=plan[key+'_sha256']:raise ValueError(f'Changed {key}')
    for arm in plan['arms']:
        out=root/arm['name']
        if not (out/'report.json').exists():
            command=[sys.executable,'-u','-m','scripts.train_human_camera_policy','--encoder',plan['encoder'],
                     '--reference',str(root/'reference.bin'),'--out',str(out),'--steps',str(arm['updates']),
                     '--natural-movement-frequency','--balance-titles','--balance-difficulties','--cpu-replay','--file-backed-replay',
                     '--train',*[str(paths[h]) for h in plan[arm['training_set']]],'--held',*[str(paths[h]) for h in plan['development']]]
            subprocess.run(command,check=True)
        report=json.loads((out/'report.json').read_bytes())
        if checksum(out/'policy.ts')!=report['torchscript_sha256'] or sum(report['observed_sampled_windows_by_game_rank'].values())!=arm['updates']*64:
            raise ValueError('Unexpected trained actor')
        print('Finished',arm['name'],report['torchscript_sha256'],flush=True)


if __name__=='__main__':main()
