"""Package the exact registered learner inputs in independently verified title archives."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

from scripts.checkpoint_storage import digest


def read(path):return json.loads(path.read_bytes())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True,type=Path);p.add_argument('--public',required=True,type=Path)
    p.add_argument('--repo',required=True);p.add_argument('--tag',required=True)
    args=p.parse_args();d=args.data.resolve();dest=args.public.resolve()/'four-title-complete-epochs'
    plan=read(dest/'protocol.json')
    b=read(d/'human-train-six-complete-credits/later-train-complete-roster-frozen.json');dev=read(d/'spatial-policy-study/protocol.json')
    fresh=read(d/'human-retail17-wave2-held/held-original-synchronized-frozen.json')
    paths={v['archive_sha256']:d/v['path'] for v in b['prior47_train']+b['qualified']+fresh['qualified']}
    paths.update({v['sha256']:d/v['file'] for v in dev['split']['held']})
    packages=defaultdict(list)
    for role in ('train','development'):
        for v in plan[role]:
            src=paths[v['sha256']]
            name=role+'-'+v['title'].lower();internal=Path('bundle')/v['file']
            packages[name].extend([(src,internal,v['sha256']),(src.with_suffix('.json'),internal.with_suffix('.json'),v['metadata_sha256'])])
    design=read(d/'ranked-multigame-sft/expanded-four-title-synchronized-fit.json')
    packages['encoder-and-registration']=[(d/design['parent']/'camera-policy.pt',Path('bundle/encoder.pt'),plan['encoder_sha256']),
                                         (dest/'protocol.json',Path('bundle/protocol.json'),digest(dest/'protocol.json'))]
    records={}
    for name,files in packages.items():
        final=dest/(name+'.zip')
        receipt=final.with_suffix('.release.json')
        if not receipt.exists():
            if not final.exists():
                tmp=final.with_suffix('.part')
                with zipfile.ZipFile(tmp,'w',zipfile.ZIP_STORED) as archive:
                    for src,internal,h in files:
                        if digest(src)!=h:raise ValueError('Changed input '+str(src))
                        archive.write(src,internal.as_posix())
                tmp.replace(final)
            with zipfile.ZipFile(final) as archive:
                if len(archive.namelist())!=len(files):raise ValueError('Incomplete stage archive')
                for _,internal,h in files:
                    with archive.open(internal.as_posix()) as fp:
                        if hashlib.file_digest(fp,'sha256').hexdigest()!=h:raise ValueError('Mismatched packaged camera input')
            subprocess.run([sys.executable,'-m','scripts.publish_original_record_bundles','--public',str(args.public),
                            '--repo',args.repo,'--tag',args.tag,str(final)],check=True)
        records[name]=read(receipt)
        # The original stage files and a remotely verified archive remain available.
        if final.exists() and digest(final)==records[name]['sha256']:final.unlink()
    inputs=dest/'training-data.md'
    lines=['# Reproducing the four-title camera training run', '',
           'Extract the archives under the same directory. The resulting `bundle` contains the 74 training stages, 18 development stages, original stage metadata, the registered initialization and the protocol. The trainer verifies both data and metadata hashes before fitting.', '',
           '```sh','python -m scripts.train_camera_full_epochs --bundle bundle --out fit','```', '',
           '| Inputs | Bytes | SHA-256 |','|---|---:|---|']
    for name,r in records.items():lines.append(f"| [{name}]({r['asset']}) | {r['bytes']:,} | `{r['sha256']}` |")
    inputs.write_text('\n'.join(lines)+'\n',encoding='utf8')


if __name__=='__main__':main()
