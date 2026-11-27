"""Publish selected sealed native logs as release assets and verify the remote byte digests."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from scripts.checkpoint_storage import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public',type=Path,required=True)
    parser.add_argument('--repo',required=True)
    parser.add_argument('--tag',required=True)
    parser.add_argument('files',nargs='+',type=Path)
    args=parser.parse_args()
    base=args.public.resolve()
    def assets():
        data=subprocess.check_output(['gh','release','view',args.tag,'--repo',args.repo,'--json','assets'])
        return {r['name']:r for r in json.loads(data)['assets']}
    for file in args.files:
        file=file.resolve()
        relative=file.relative_to(base)
        bundle=file.suffix=='.zip' and len(relative.parts)==2
        film=file.suffix=='.mp4' and len(relative.parts)==3 and relative.parts[1] in ('evaluation','first-phase-recordings')
        if not (bundle or film):raise ValueError(f'Expected a study record bundle or verified evaluation film {file}')
        if film:
            receipt=json.loads(file.with_suffix('.json').read_bytes())
            field='video_sha256' if relative.parts[1]=='first-phase-recordings' else 'sha256'
            if receipt[field]!=digest(file):raise ValueError('Film bytes differ from the native rendering receipt')
        name=relative.parts[0]+'-'+file.name
        checksum=digest(file)
        for attempt in range(4):
            try:
                remote=assets().get(name)
                if remote is None:
                    with tempfile.TemporaryDirectory(dir=base.parent.parent.parent) as tmp:
                        linked=Path(tmp)/name
                        os.link(file,linked)
                        subprocess.run(['gh','release','upload',args.tag,str(linked),'--repo',args.repo],check=True)
                    remote=assets()[name]
                break
            except (subprocess.SubprocessError,KeyError):
                if attempt==3:raise
                time.sleep(10)
        if remote['size']!=file.stat().st_size or remote['digest']!='sha256:'+checksum:
            raise ValueError(f'Mismatched published bytes for {name}')
        out=file.with_suffix('.release.json')
        out.write_text(json.dumps(dict(asset=remote['url'],sha256=checksum,bytes=remote['size'],
                                      original_filename=file.name),indent=2)+'\n',encoding='utf8')
        print('Verified',name,remote['size'],flush=True)


if __name__=='__main__':
    main()
