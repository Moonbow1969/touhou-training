"""Run a matched-compute human policy SFT with frozen, synchronized later story stages."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze

def read(path):return json.loads(path.read_bytes())


def main():
    arg=argparse.ArgumentParser(description=__doc__)
    for name in ('data','reference'):arg.add_argument('--'+name,type=Path,required=True)
    a=arg.parse_args();d=a.data.resolve();reference=a.reference.resolve()
    root=d/'four-title-later-stage-sft';root.mkdir(exist_ok=True)
    src=d/'human-train-six-complete-credits/later-train-first-21-frozen.json'
    if digest(src)!='5707d139f0cd622f9de6c5974df35342ba365ed7df59516769cc6efdfc768554':
        raise ValueError('Changed incremental source registration')
    sources=read(src);old=d/sources['prior47_report'];before=read(old)
    if digest(old)!=sources['prior47_report_sha256'] or before['torchscript_sha256']!=sources['prior47_policy_sha256']:
        raise ValueError('Prior comparison actor changed')
    base=d/'ranked-multigame-sft/expanded-four-title-synchronized-fit.json';design=read(base)
    encoder=d/design['parent']/'camera-policy.pt'
    train=sources['prior47_train']+sources['qualified']
    held=read(d/design['base_split'])['split']['held']
    if len(train)!=68 or {r['archive_sha256'] for r in sources['prior47_train']}!={r['sha256'] for r in before['train']}:
        raise ValueError('Unmatched baseline population')
    if len({(r['replay_sha256'],r['stage']) for r in train})!=68 or any(r['split']!='train' for r in train):
        raise ValueError('Duplicate or non-training original stage')
    if sources['prior_camera_held_overlap'] or sources['new_TH17_held_overlap']:raise ValueError('Unseen pilot overlaps')
    paths=[d/r['path'] for r in train]
    if any(digest(f)!=r['archive_sha256'] for f,r in zip(paths,train)):raise ValueError('Changed training cameras')
    if {r['sha256'] for r in held}!={r['sha256'] for r in before['held']} or any(digest(d/r['file'])!=r['sha256'] for r in held):
        raise ValueError('Changed evaluation sources')
    if digest(encoder)!=before['encoder'] or before['steps']!=3000 or before['head_label_smoothing']!=[.025]*4:
        raise ValueError('Changed initial parameters or control training recipe')
    freeze(root/'protocol.json',dict(source=digest(src),initial_encoder=digest(encoder),baseline_actor=sources['prior47_policy_sha256'],
        original_fit_registration=digest(base),reference_policy=digest(reference),updates=3000,batch=64,
        prior_stages=47,added_stages=21,available_windows=231511,
        design='Add later-stage synchronized human cameras on TH10, TH12 and TH13 to the identical four-title SFT starting weights. Preserve equal title and difficulty sampling, 3000 optimizer updates, natural movement labels and all four action heads. New stages come from five existing training authors selected for recorded Bomb events. The increment has no TH13 Lunatic stages. The test measures a change in the training distribution at fixed compute.',
        held='Compare against the previous 47-stage SFT on the same ten held stage files. Report this reused development set separately from the four reserved TH17 pilots. Record all movement transitions and Bomb onsets. These behavioral comparisons require a separate original-credit test to establish survival.'))
    out=root/'policy'
    if not (out/'report.json').is_file():
        subprocess.run([sys.executable,'-u','-m','scripts.train_human_camera_policy','--encoder',str(encoder),
            '--reference',str(reference),'--out',str(out),'--steps','3000','--cpu-replay','--natural-movement-frequency',
            '--balance-titles','--balance-difficulties','--bomb-label-smoothing','.025','--train',*map(str,paths),
            '--held',*[str(d/r['file']) for r in held]],check=True)
    report=read(out/'report.json')
    if sum(r['decisions'] for r in report['train'])!=231511 or {r['sha256'] for r in report['train']}!={r['archive_sha256'] for r in train}:
        raise ValueError('Learner source observations do not match the registered union')
    print('Matched four-title SFT complete',report['torchscript_sha256'],flush=True)
    subprocess.run([sys.executable,'-m','scripts.compare_human_camera_policies','--models',str(old.parent),str(out),
        '--model-names','prior47','later68','--held',*[str(d/r['file']) for r in held],'--out',str(root/'repeated-dev')],check=True)


if __name__=='__main__':
    main()
