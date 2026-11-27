"""Compare two human SFT objectives targeting active movement changes and rare Bomb onsets."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze


def read(path):return json.loads(path.read_bytes())
def wait(path):
    while not path.is_file():time.sleep(15)
def run(module,*args):
    subprocess.run([sys.executable,'-u','-m','scripts.'+module,*map(str,args)],check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ('data','reference'):parser.add_argument('--'+k,required=True,type=Path)
    parser.add_argument('--arm',choices=('bomb','movement'),required=True)
    a=parser.parse_args();d=a.data.resolve();root=d/'four-title-human-events';root.mkdir(exist_ok=True)
    earlier=d/'four-title-later-stage-sft';parent=earlier/'policy'
    manifest=d/'human-train-six-complete-credits/later-train-first-21-frozen.json';m=read(manifest)
    paths=[d/r['path'] for r in m['prior47_train']+m['qualified']]
    old=read(parent/'report.json')
    if {digest(f) for f in paths}!={r['sha256'] for r in old['train']} or digest(parent/'policy.ts')!=old['torchscript_sha256']:
        raise ValueError('Expected exactly the same human four-title population and checkpoint')
    design=read(d/'ranked-multigame-sft/expanded-four-title-synchronized-fit.json')
    dev=[d/r['file'] for r in read(d/design['base_split'])['split']['held']]
    prospective=d/'unseen-later-stage-author-challenge/prospective-six-players-frozen.json'
    freeze(root/f'{a.arm}-protocol.json',dict(parent=digest(parent/'policy.ts'),source_manifest=digest(manifest),prospective=digest(prospective),
        comparison=('Double the supervised direction loss for an expert starting a movement or turning to another moving direction; normalize the multiplier by its mean within replay. Keep initialization, 3000 updates, samples, label smoothing and all four action heads the same as the 68-stage control. Primary future outcome is the equal-player mean of start and turn negative log likelihood, also report stop and overall accuracy.'
                    if a.arm=='movement' else
                    'Use the established frozen-feature case-weighted Bomb onset logistic objective, with fixed 20x positive weight and prior odds correction. Same 68 human stages and three other action heads as the parent. Primary future outcome is average precision averaged over held pilots with at least one true onset. Report aggregate true edges, false alerts, calibration and all zero-onset stages.'),
        validation='The four TH17 players and original development set informed this follow-up. Treat those comparisons as repeated development. Six new TH10/12/13 players and stages were registered before fitting. Use all synchronized sources from that roster and retain failures. A native action ablation is required to attribute a gameplay improvement.'))
    out=root/a.arm
    if not (out/'report.json').exists():
        if a.arm=='bomb':
            run('fit_human_bomb_onset_head','--parent',parent,'--train',*paths,'--out',out)
        else:
            # Avoid overlapping the larger host-side human replay bank with the already scheduled PPO generations.
            wait(d/'six-game-return-horizon/six-return/policy/report.json')
            wait(d/'later-human-online/policy/report.json')
            encoder=d/design['parent']/'camera-policy.pt'
            if digest(encoder)!=old['encoder']:raise ValueError('Changed control initialization')
            run('train_human_camera_policy','--encoder',encoder,'--reference',a.reference,'--out',out,'--steps',3000,
                '--cpu-replay','--file-backed-replay','--natural-movement-frequency','--active-change-weight',2,'--balance-titles','--balance-difficulties',
                '--bomb-label-smoothing','.025','--train',*paths,'--held',*dev)
    sources=read(d/'human-retail17-wave2-held/held-original-synchronized-frozen.json')['qualified']
    def compare(files,output):
        if a.arm=='bomb':run('score_human_bomb_onsets','--models',parent,out,'--held',*files,'--out',output/'onsets.json')
        else:run('compare_human_camera_policies','--models',parent,out,'--model-names','natural68','active-change68','--held',*files,'--out',output)
    compare([d/v['path'] for v in sources],out/'repeated-th17')
    # The native collector writes this qualified roster only after adjudicating all six allocated stages.
    held=d/'unseen-later-stage-author-challenge/held-original-synchronized-frozen.json'
    wait(held);results=read(held);authorized=read(prospective)
    expected={(v['sha256'],v['reserved_stage']) for v in authorized['sources']}
    assigned=[(v['replay_sha256'],v['stage']) for v in results['qualified']+results['excluded']]
    if len(expected)!=6 or len(assigned)!=6 or set(assigned)!=expected or any(v['split']!='held' for v in results['qualified']):
        raise ValueError('Unregistered or missing held allocation')
    final=[]
    for v in results['qualified']:
        path=d/v['path']
        if digest(path)!=v['archive_sha256']:raise ValueError('Changed held-stage bytes')
        final.append(path)
    compare(final,out/'prospective-six')


if __name__=='__main__':
    main()
