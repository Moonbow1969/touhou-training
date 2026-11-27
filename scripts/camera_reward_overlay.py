"""Archive changed per-action rewards without copying the original camera frames."""
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
from scripts.checkpoint_storage import digest


def index(folder):
    records=json.loads((folder/'credits.json').read_bytes())
    result={(r['title'],r['seed']):r for r in records}
    if len(result)!=len(records):raise ValueError('Duplicate credit')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['export','apply'])
    p.add_argument('--original',type=Path,required=True);p.add_argument('--variant',type=Path,required=True)
    p.add_argument('--overlay',type=Path,required=True);a=p.parse_args()
    original=index(a.original);changed=[]
    if a.mode=='export':
        variant=index(a.variant)
        if original.keys()!=variant.keys():raise ValueError('Different credit cohort')
        a.overlay.mkdir(parents=True,exist_ok=True)
        for k,row in original.items():
            after=variant[k];source=a.original/row['file'];target=a.variant/after['file']
            if digest(source)!=row['sha256'] or digest(target)!=after['sha256']:raise ValueError('Changed source archive')
            if row['sha256']==after['sha256']:continue
            with np.load(source,allow_pickle=False) as old,np.load(target,allow_pickle=False) as new:
                for name in old.files:
                    if name!='reward' and not np.array_equal(old[name],new[name]):raise ValueError(f'Modified nonreward field {name}')
                additions={name:new[name] for name in new.files if name=='reward' or name not in old.files}
                output=a.overlay/row['file'];np.savez_compressed(output,**additions)
                changed.append(dict(title=k[0],seed=k[1],file=row['file'],source_sha256=row['sha256'],overlay_sha256=digest(output),
                                    key_order=new.files,variant_sha256=after['sha256'],changed_reward_steps=int(np.count_nonzero(old['reward']!=new['reward']))))
        (a.overlay/'overlay.json').write_text(json.dumps(dict(original_manifest_sha256=digest(a.original/'credits.json'),changes=changed),indent=2)+'\n')
    else:
        meta=json.loads((a.overlay/'overlay.json').read_bytes());a.variant.mkdir(parents=True,exist_ok=True)
        if digest(a.original/'credits.json')!=meta['original_manifest_sha256']:raise ValueError('Wrong baseline manifest')
        additions={(r['title'],r['seed']):r for r in meta['changes']}
        if not set(additions).issubset(original):raise ValueError('Unknown overlay seed')
        for k,row in original.items():
            source=a.original/row['file'];dest=a.variant/row['file']
            if dest.exists():raise FileExistsError(dest)
            if digest(source)!=row['sha256']:raise ValueError('Changed original camera file')
            if k not in additions:
                try:os.link(source,dest)
                except OSError:shutil.copyfile(source,dest)
                continue
            r=additions[k];overlay=a.overlay/r['file']
            if row['sha256']!=r['source_sha256'] or digest(overlay)!=r['overlay_sha256']:raise ValueError('Overlay does not match credit')
            with np.load(source,allow_pickle=False) as old,np.load(overlay,allow_pickle=False) as patch:
                np.savez_compressed(dest,**{name:patch[name] if name in patch else old[name] for name in r['key_order']})
            row['sha256']=digest(dest)
            if row['sha256']!=r['variant_sha256']:raise ValueError('Rebuilt learner data differs from registered variant')
        (a.variant/'credits.json').write_text(json.dumps(list(original.values()),indent=2)+'\n')


if __name__=='__main__':
    main()
