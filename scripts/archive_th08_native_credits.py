"""Archive qualified TH08 sources in bounded bundles with the hashes used for score and optimization."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from scripts.checkpoint_storage import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','public'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--repo',required=True);p.add_argument('--tag',required=True)
    a=p.parse_args();root=a.root.resolve();public=a.public.resolve();target=public/root.name
    target.mkdir(exist_ok=True,parents=True)
    files_to_save=('protocol.json','receipt.json','timing.jsonl','behavior.npz','native-observers.npz',
                   'chosen-rank.png','first-policy-frame.png','terminal.png','timeout.png','technical-stop.png')
    seeds=json.loads((root/'protocol.json').read_bytes())['th08_train_seeds']
    for j in range(0,len(seeds),8):
        slots=seeds[j:j+8]
        selection=[root/'selected-training'/f'{s}.json' for s in slots]
        while not all(f.is_file() for f in selection):
            time.sleep(12)
        archive=target/f'th08-physical-credits-{j+1:02d}-{j+len(slots):02d}.zip'
        if not archive.exists():
            tmp=archive.with_suffix('.inprogress')
            members={}
            def add(path):
                if path.is_file():members[path.relative_to(root).as_posix()]=path
            for f in selection:
                add(f);chosen=json.loads(f.read_bytes())
                for label in ('qualified_source','original_source'):
                    native=root/chosen[label]
                    for name in files_to_save:add(native/name)
                    if label=='original_source':
                        if digest(native/'receipt.json')!=chosen['original_receipt_sha256']:
                            raise ValueError('Changed original source')
                for rejected in chosen['rejected_attempts']:
                    dest=root/'native08'/f'train-five-ppo2-{rejected["seed"]}'
                    for name in files_to_save:add(dest/name)
                row=chosen['record']
                inputs=root/'th08-training'/row['file']
                if digest(inputs)!=row['sha256']:raise ValueError('Changed learner input')
                add(inputs)
            for name in ('protocol.json','technical-menu-amendment.json'):add(root/name)
            with zipfile.ZipFile(tmp,'w',allowZip64=True) as z:
                for name,path in sorted(members.items()):
                    z.write(path,name,compress_type=zipfile.ZIP_DEFLATED if path.suffix in ('.json','.jsonl') else zipfile.ZIP_STORED)
            tmp.replace(archive)
        subprocess.run([sys.executable,'-m','scripts.publish_original_record_bundles',
                        '--public',str(public),'--repo',a.repo,'--tag',a.tag,str(archive)],check=True)
        print('Verified original observer and optimizer source',archive.name,archive.stat().st_size,flush=True)


if __name__=='__main__':
    main()
