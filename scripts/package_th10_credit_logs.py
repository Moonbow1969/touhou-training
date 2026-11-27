"""Package native endpoints and policy traces, with a checksum of each original image stream."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def package(out,directories):
    entries=[];out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_suffix('.inprogress')
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as archive:
        for folder in directories:
            arm=folder.name;files=sorted(folder.glob('*/report.json'))
            if not files:raise ValueError(f'No sealed credits in {folder}')
            for file in files:
                r=json.loads(file.read_bytes());credit=file.parent
                if not r['terminal'] or r['continues']!=0 or r['title']!=10:raise ValueError(f'Incomplete native credit in {file}')
                checks={}
                for name in ['report.json','behavior.npz','native.csv.gz','endpoint.png']:
                    data=(credit/name).read_bytes()
                    checked=sha(data)
                    if name=='behavior.npz':assert checked==r['behavior_sha256']
                    if name=='endpoint.png':assert checked==r['terminal_screenshot_sha256']
                    checks[name]=checked
                    z=zipfile.ZipInfo(f'{arm}/{credit.name}/{name}');z.compress_type=zipfile.ZIP_DEFLATED
                    archive.writestr(z,data)
                entries.append(dict(arm=arm,host=r['host'],seed=r['seed'],steps=r['steps'],policy_sha256=r['policy_sha256'],
                    original_pixel_npz_sha256=sha((credit/'original-pixels.npz').read_bytes()),
                    native_outcome=r['native_outcome'],files=checks))
        z=zipfile.ZipInfo('index.json');z.compress_type=zipfile.ZIP_DEFLATED
        archive.writestr(z,json.dumps(dict(credits=entries,pixels='The separate original-pixels.npz source camera archives are identified by their whole-file hash.'),indent=2)+'\n')
    with zipfile.ZipFile(tmp) as archive:
        if archive.testzip() is not None:raise IOError('Unreadable archive member')
    tmp.replace(out)
    print(json.dumps(dict(credits=len(entries),sha256=sha(out.read_bytes()),bytes=out.stat().st_size)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('directories',nargs='+',type=Path)
    args=p.parse_args();package(args.out,args.directories)
