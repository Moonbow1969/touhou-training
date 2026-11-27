"""Introduce synchronized TH11 human cameras into a matched four- and five-title SFT comparison."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze


def read(path):return json.loads(path.read_bytes())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    a=p.parse_args();d=a.data.resolve();root=d/'first-five-title-retail-sft';root.mkdir(exist_ok=True)
    base=d/'human-train-six-complete-credits/later-train-complete-roster-frozen.json'
    future=d/'th11-expansion/retail-original-camera/train-first-hard-lunatic-original-synchronized-frozen.json'
    pilot=d/'th11-expansion/retail-original-camera/held-qualified-01-stages-frozen.json'
    dev=d/'spatial-policy-study/protocol.json';extra=d/'human-retail17-wave2-held/held-original-synchronized-frozen.json'
    prior=d/'four-title-later-stage-sft/policy';enc=d/read(d/'ranked-multigame-sft/expanded-four-title-synchronized-fit.json')['parent']/'camera-policy.pt'
    if digest(enc)!=read(prior/'report.json')['encoder']:raise ValueError('Different initialization')
    protocol=root/'protocol.json'
    freeze(protocol,dict(base=digest(base),common_initialization=digest(enc),reference=digest(a.reference),
        previous_development=digest(dev),repeated_th17=digest(extra),first_th11_development=digest(pilot),
        train_addition='First qualified Hard and Lunatic TH11 retail Stage 1 sources from separate training players. Require native build, input synchronization and replay checksum evidence. Source assignment follows capture order only. Keep every held author isolated.',
        recipe='Four-title control includes all 74 sealed TH10/12/13/17 human stages. Five-title arm adds one synchronized TH11 Hard stage and one Lunatic stage. Use identical encoder, ordinary four-head objective, training seed, 3000 updates of 64, equal title/rank mass, replay-balanced sampling, movement importance correction, flip and brightness augmentation. Adding a fifth title redistributes the fixed sampling budget.',
        outcome='Compare all movements, starts, turns, stops and Bombs on the same developer cohorts, with separate reports by title and player. The first TH11 held credit is one Lunatic pilot. Later reserved TH11 pilots are an independent test. Report available and drawn examples per title and rank.') )
    b=read(base);common=b['prior47_train']+b['qualified']
    if len(common)!=74:raise ValueError('Unexpected base data size')
    tests=[(d/v['file'],v['sha256']) for v in read(dev)['split']['held']]
    tests.extend((d/v['path'],v['archive_sha256']) for m in (extra,pilot) for v in read(m)['qualified'])
    if any(digest(file)!=sha for file,sha in tests):raise ValueError('Changed held camera')
    for name in ('four','five'):
        rows=common
        if name=='five':
            while not future.exists():time.sleep(15)
            new=read(future)['qualified']
            if len(new)!=2 or {v['difficulty'] for v in new}!={'Hard','Lunatic'} or len({v['player_group'].casefold() for v in new})!=2:
                raise ValueError('Expected two distinct TH11 training players and both ranks')
            for v in new:
                if v['title']!='TH11' or v['split']!='train' or digest(d/v['original_sync_evidence_path'])!=v['original_sync_evidence_sha256']:
                    raise ValueError('Missing native TH11 human evidence')
            rows=common+new
            freeze(root/'added-th11.json',dict(source=digest(future),players=new))
        files=[d/v['path'] for v in rows]
        if any(digest(file)!=v['archive_sha256'] for file,v in zip(files,rows)):raise ValueError('Changed training archive')
        dest=root/name
        if not (dest/'report.json').exists():
            subprocess.run([sys.executable,'-u','-m','scripts.train_human_camera_policy','--encoder',str(enc),'--reference',str(a.reference),
                '--out',str(dest),'--steps','3000','--natural-movement-frequency','--balance-titles','--balance-difficulties','--cpu-replay','--file-backed-replay',
                '--train',*map(str,files),'--held',*[str(path) for path,_ in tests]],check=True)
        result=read(dest/'report.json')
        if len(result['train'])!=len(rows) or digest(dest/'policy.ts')!=result['torchscript_sha256'] or sum(result['observed_sampled_windows_by_game_rank'].values())!=192000:
            raise ValueError('Unexpected completed fit')
        print('Fitted',name,len(rows),result['observed_sampled_windows_by_game_rank'],result['torchscript_sha256'],flush=True)
    subprocess.run([sys.executable,'-u','-m','scripts.compare_human_camera_policies','--models',str(root/'four'),str(root/'five'),
                    '--model-names','four','five','--held',*[str(path) for path,_ in tests],'--out',str(root/'development')],check=True)


if __name__=='__main__':main()
