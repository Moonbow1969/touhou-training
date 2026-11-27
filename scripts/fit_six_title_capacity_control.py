"""Train equal-data CNN and spatial-attention policies on six human camera titles."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze


def read(path):return json.loads(path.read_bytes())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data','reference','public'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();d=a.data.resolve()
    root=d/'six-title-camera-capacity';root.mkdir(exist_ok=True)
    pub=a.public.resolve()/root.name;pub.mkdir(exist_ok=True)
    source=d/'six-title-human-camera-th16';original=read(source/'policy/report.json')
    common=read(d/'human-train-six-complete-credits/later-train-complete-roster-frozen.json')
    rows=common['prior47_train']+common['qualified']+read(d/'first-five-title-retail-sft/added-reviewed-th11.json')['players']+read(source/'added-th16.json')['players']
    train=[d/v['path'] for v in rows]
    held=[d/v['file'] for v in read(d/'spatial-policy-study/protocol.json')['split']['held']]
    for name in ('human-retail17-wave2-held/held-original-synchronized-frozen.json','th11-expansion/retail-original-camera/held-qualified-01-stages-frozen.json'):
        held.extend(d/v['path'] for v in read(d/name)['qualified'])
    if any([digest(f) for f in files]!=[e['sha256'] for e in original[key]] for files,key in ((train,'train'),(held,'held'))):
        raise ValueError('Different original sources')
    enc=d/read(d/'ranked-multigame-sft/expanded-four-title-synchronized-fit.json')['parent']/'camera-policy.pt'
    ref=a.reference.resolve()
    if digest(enc)!=original['encoder'] or digest(ref)!=read(d/'first-five-title-retail-sft/protocol.json')['reference']:raise ValueError('Different initializing model')
    protocol=root/'protocol.json'
    freeze(protocol,dict(six_title_sft_report=digest(source/'policy/report.json'),train_stages=len(train),train_windows=sum(v['decisions'] for v in original['train']),
           comparison='Fit the existing convolutional encoder and its extended spatial attention model from the same encoder. The attention residual has zero output at initialization. The matched 3,000-update fits must share every training index, flip and brightness factor. All four action heads are learned. Archive both models whatever the result.',
           outcomes='Compare expert movements, starts and turns per game on 19 development stages. Measure the three repeated TH11 players separately. The fresh TH16 held roster supplies a prospective test after native qualification. Original survival requires independent gameplay.'))
    shutil.copy2(protocol,pub/protocol.name)
    fits={}
    for name,extra in [('cnn',[]),('attention',['--spatial-attention'])]:
        out=root/name
        if not (out/'report.json').is_file():
            subprocess.run([sys.executable,'-u','-m','scripts.train_human_camera_policy','--encoder',str(enc),'--reference',str(ref),'--steps','3000',
                '--natural-movement-frequency','--balance-titles','--balance-difficulties','--cpu-replay','--file-backed-replay','--audit-draws',*extra,
                '--out',str(out),'--train',*map(str,train),'--held',*map(str,held)],check=True)
        v=read(out/'report.json')
        if digest(out/'policy.ts')!=v['torchscript_sha256'] or digest(out/'camera-policy.pt')!=v['sft'] or len(v['train'])!=86:raise ValueError('Unverified fit')
        fits[name]=v
        (pub/name).mkdir(exist_ok=True)
        for f in ('report.json','camera-policy.pt','policy.ts','loss.jsonl'):shutil.copy2(out/f,pub/name/f)
        print('Fitted',name,v['deployed_actor_parameters'],v['ordered_draw_flip_brightness_sha256'],v['torchscript_sha256'],flush=True)
    if fits['cnn']['ordered_draw_flip_brightness_sha256']!=fits['attention']['ordered_draw_flip_brightness_sha256']:
        raise ValueError('Fit pair has different training examples or augmentations')
    repeats=read(d/'th11-expansion/retail-clear-boundary-review/external-held-original-synchronized-frozen.json')['qualified']
    for cohort,tests in [('development',held),('repeated-th11',[d/v['path'] for v in repeats])]:
        subprocess.run([sys.executable,'-m','scripts.compare_human_camera_policies','--models',str(root/'cnn'),str(root/'attention'),
                        '--model-names','cnn','attention','--held',*map(str,tests),'--out',str(root/cohort)],check=True)
        shutil.copytree(root/cohort,pub/cohort,dirs_exist_ok=True)
    freeze(pub/'matched-inputs.json',dict(ordered_draw_flip_brightness_sha256=fits['cnn']['ordered_draw_flip_brightness_sha256'],presentations=192000,
        models={k:dict(checkpoint=v['torchscript_sha256'],parameters=v['deployed_actor_parameters']) for k,v in fits.items()}))


if __name__=='__main__':main()
