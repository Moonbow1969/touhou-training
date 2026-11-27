"""Compare equal-budget retail rehearsal and exact-build human replay augmentation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from scripts.checkpoint_storage import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data','plan','reference','out'):
        p.add_argument('--'+key,required=True,type=Path)
    p.add_argument('--manifests',nargs='+',type=Path,required=True)
    a=p.parse_args()
    data=a.data.resolve()
    plan=json.loads(a.plan.read_bytes())
    scale=json.loads((data/plan['scale_manifest']).read_bytes())
    base=json.loads((data/plan['base_split']).read_bytes())['split']
    def resolve(path,sha):
        f=(data/path).resolve()
        f.relative_to(data)
        if digest(f)!=sha:raise ValueError(f'Changed asset {path}')
        return f
    ordinary=[resolve(r['file'],r['sha256']) for r in base['train']]
    ordinary.extend(resolve(r['path'],r['archive_sha256']) for r in scale['records'])
    held=[resolve(r['file'],r['sha256']) for r in base['held']]
    expected={(r['title'],r['split'],r['replay_sha256'],r['stage']):r['player_group'] for r in plan['trial_stages']}
    found={}
    trial={'train':[],'held':[]}
    players={s:set() for s in trial}
    for manifest in a.manifests:
        cohort=json.loads(manifest.read_bytes())
        for r in cohort['qualified']:
            key=(r['title'],r['split'],r['replay_sha256'],r['stage'])
            if r['split']!=cohort['split'] or r['player_group']!=expected.get(key) or key in found:
                raise ValueError(f'Unexpected trial source {key}')
            f=resolve(r['path'],r['archive_sha256'])
            side=json.loads(f.with_suffix('.json').read_bytes())
            if side['player_group']!=r['player_group'] or side['replay_sha256']!=r['replay_sha256']:
                raise ValueError('Trial sidecar differs')
            trial[r['split']].append(f)
            found[key]=f
            players[r['split']].add(r['player_group'])
    if expected.keys()!=found.keys():raise ValueError(f'Incomplete original replays ({len(found)}/{len(expected)})')
    if players['train'] & players['held']:raise ValueError('Trial pilot leakage')
    retail_players={r['player_group'] for r in scale['records']} | {r['player'] for s in ('train','held') for r in base[s]}
    if (players['train'] | players['held']) & retail_players:raise ValueError('Pilot appears in both source corpora')
    parent=data/plan['parent']
    policy=json.loads((parent/'report.json').read_bytes())
    weights=parent/'camera-policy.pt'
    if digest(weights)!=policy['sft'] or policy['spatial_attention']:raise ValueError('Unexpected parent CNN')
    if {r['sha256'] for r in policy['train']}!={digest(f) for f in ordinary}:raise ValueError('Retail split mismatch')
    a.out.mkdir(parents=True,exist_ok=True)
    proof=dict(plan_sha256=digest(a.plan),parent_sha256=digest(weights),train_source_shas=[digest(f) for f in ordinary],
               exact_train_source_shas=[digest(f) for f in trial['train']],exact_held_source_shas=[digest(f) for f in trial['held']],
               manifests={m.name+':'+m.parent.name:digest(m) for m in a.manifests})
    path=a.out/'input-checksums.json'
    if path.exists() and json.loads(path.read_bytes())!=proof:raise ValueError('Existing fine-tune registration differs')
    path.write_text(json.dumps(proof,indent=2)+'\n',encoding='utf8')
    for arm in ('retail-rehearsal','trial-augmented'):
        dest=a.out/arm
        if (dest/'report.json').is_file():
            result=json.loads((dest/'report.json').read_bytes())
            if result['encoder']!=digest(weights) or result['steps']!=plan['updates'] or result['torchscript_sha256']!=digest(dest/'policy.ts'):
                raise ValueError('Fine-tune checkpoint differs')
            continue
        train=ordinary+(trial['train'] if arm=='trial-augmented' else [])
        subprocess.run([sys.executable,'-u','-m','scripts.train_human_camera_policy','--encoder',str(weights),
            '--reference',str(a.reference),'--out',str(dest),'--steps',str(plan['updates']),
            '--cpu-replay','--natural-movement-frequency','--balance-titles',
            '--train',*map(str,train),'--held',*map(str,held+trial['held'])],check=True)


if __name__=='__main__':
    main()
