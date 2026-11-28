"""Publish completed source archives with independent GitHub digest checks."""
import argparse,hashlib,json
from pathlib import Path
import subprocess

def gh(*args):
    return subprocess.check_output(['gh',*args],text=True,encoding='utf-8')
def remote_assets(repo,tag):
    release=json.loads(gh('api','repos/'+repo+'/releases/tags/'+tag))
    return {r['name']:r for r in release['assets']}
def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--repo',required=True);p.add_argument('--tag',required=True)
    args=p.parse_args();index=json.loads((args.source/'archive-manifest.json').read_bytes())
    for f in index['archives']:
        path=args.source/f['name']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=f['sha256']:raise ValueError('Local archive changed')
        assets=remote_assets(args.repo,args.tag)
        if f['name'] not in assets:
            gh('release','upload',args.tag,str(path),'--repo',args.repo)
            assets=remote_assets(args.repo,args.tag)
        remote=assets[f['name']]
        if remote['digest']!='sha256:'+f['sha256'] or remote['size']!=f['bytes']:
            raise ValueError('Remote source hash or byte count changed')
        f.update(url=remote['browser_download_url'],remote_digest=remote['digest'])
        print(f['name'],f['bytes'],f['sha256'],flush=True)
    final=args.source/'published-sources.json'
    tmp=final.with_suffix('.pending');tmp.write_text(json.dumps(index,indent=2)+'\n');tmp.replace(final)
if __name__=='__main__':main()
