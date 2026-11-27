"""Register named, player-disjoint retail TH11 replay sources before their first native camera capture."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import unicodedata

from replays.build_corpus import player_group


def fingerprint(path):
    return sha256(path.read_bytes()).hexdigest()


def group(name):
    return player_group(unicodedata.normalize('NFKC', name).strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    args = parser.parse_args()
    root = args.data.resolve()
    directory = root / 'th11-expansion'
    file = directory / 'action-source-manifest.json'
    rows = json.loads(file.read_bytes())['records']
    fit, dev = set(), set()
    parents = {}
    for path in ('scaled-human-sft/new-64/policy/report.json', 'ranked-multigame-sft/four-title/report.json',
                 'exact-build-human/43-stage-pair/trial-augmented/report.json'):
        report = json.loads((root/path).read_bytes())
        parents[path] = fingerprint(root/path)
        fit.update(group(r['group']) for r in report['train'])
        dev.update(group(r['group']) for r in report['held'])
    exclusions = []
    chosen = []
    for r in sorted(rows,key=lambda v:v['sha256']):
        if r['difficulty'] not in ('Hard','Lunatic'):
            continue
        key=group(r['player'])
        role='train' if r['split']=='train' else 'held'
        if (role=='train' and key in dev) or (role=='held' and key in fit):
            exclusions.append(dict(source=r['name'],player=key,reason='Conflicts with prior cross-title player role'))
            continue
        chosen.append(dict(file=r['name'],sha256=r['sha256'],source=r['source'],difficulty=r['difficulty'],
                           native_version=r['version'],role=role,player=key,shot=r['shot'],
                           requested_stage=1,expected_native_stage_ticks=r['frames_by_stage']['1']))
    train={r['player'] for r in chosen if r['role']=='train'}
    held={r['player'] for r in chosen if r['role']=='held'}
    if not train or not held or train&held or any(r['native_version']!='1.00a' for r in chosen):
        raise ValueError('New TH11 camera sources violate release or pilot independence')
    result=dict(source_manifest_sha256=fingerprint(file),checked_cross_title_models=parents,
                scope='Retail TH11 1.00a native camera acquisition roster. Pilot-grouped Hard and Lunatic development. Admission requires verified original camera alignment, control words and replay synchronization. No transfer accuracy is inferred from these sources.',
                train_pilots=len(train),held_pilots=len(held),sources=chosen,excluded=exclusions)
    target=directory/'retail-camera-registration.json'
    if target.exists() and json.loads(target.read_bytes())!=result:
        raise ValueError('Existing camera registration differs')
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print('Registered TH11 retail camera sources',len(chosen),'TRAIN / HELD pilots',len(train),len(held))


if __name__=='__main__':
    main()
