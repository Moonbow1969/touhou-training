"""Pack only the preregistered original credits, preserving native replays, final screens and actor data."""
import importlib.util
import json
from pathlib import Path
import subprocess
import time
import zipfile
from hashlib import sha256

root=Path('/var/tmp/moonbow-th07-trial-261127/online')
spec=importlib.util.spec_from_file_location('native_fit',root/'online-fit.py')
fit=importlib.util.module_from_spec(spec);spec.loader.exec_module(fit)
plan=fit.plan
rows=[]
sources=[]
deadline=time.monotonic()+900
for role,seeds in [('train',plan['train']),('held',plan['held'])]:
    for seed in seeds:
        for arm in ([plan['parent_arm']] if role=='train' else [plan['parent_arm'],plan['candidate']]):
            folder=root/(f'train-{seed}' if role=='train' else f'held-{arm}-{seed}')
            while not (folder/'receipt.pending-visual.json').is_file():
                if time.monotonic()>deadline:raise RuntimeError(str(folder))
                time.sleep(1)
            if role=='held':
                _,r=fit.pack(folder,role,arm)
            else:
                while not (folder/'receipt.json').is_file():time.sleep(.5)
                r=json.loads((folder/'receipt.json').read_bytes())
            expected=plan['parent_policy_sha256'] if arm==plan['parent_arm'] else json.loads((root/'policy/report.json').read_bytes())['policy_sha256']
            assert (r['policy_sha256'],r['seed'],r['role'],r['native_opening'])==(expected,seed,role,[2,3])
            for name,digest in r['evidence_sha256'].items():assert fit.sha(folder/name)==digest
            rows.append(r)
            sources.extend(folder/name for name in r['evidence_sha256'])
            sources.extend([folder/'receipt.json',folder/'menu-1.png',folder/'menu-2.png',folder/'menu-3.png'])
            if role=='held' and seed==plan['held'][0]:
                subprocess.run(['python3',str(root.parent/'movie.py'),f'online/{folder.name}'],check=True)
                sources.append(folder/'normal-original.mp4')
for path in [root/'protocol.json',root/'policy/report.json',root/'policy/policy.ts',root/'policy/state_dict.pt']:
    sources.append(path)
fit.save(root/'complete.json',dict(credits=rows,policy_sha256=fit.sha(root/'policy/policy.ts'),
                    note='Per-seed continued-game count, misses and exact successful Bomb count are read from original scorecard PNGs.'))
sources.append(root/'complete.json')
temp=root/'native-full.partial'
with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=2) as z:
    for f in sources:
        z.write(f,f.relative_to(root),compress_type=zipfile.ZIP_STORED if f.suffix in ('.npz','.png','.mp4') else zipfile.ZIP_DEFLATED)
with zipfile.ZipFile(temp) as z:assert z.testzip() is None
path=root/'eight-original-credits.zip';temp.replace(path)
print('SEALED',path.stat().st_size,fit.sha(path),flush=True)
