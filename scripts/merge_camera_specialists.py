"""Evaluate an equal-weight parameter average of four common-parent SFT specialists."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch
from scripts.checkpoint_storage import digest
from scripts.train_replay_camera_encoder import CameraPolicy


def read(path):
    return json.loads(path.read_bytes())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    d=parser.parse_args().data.resolve()
    root=d/'specialist-vs-shared'
    design=read(root/'protocol.json')
    base=read(d/'ranked-multigame-sft/design.json')
    joint=d/'ranked-multigame-sft/four-title'
    merged=root/'merged-equal'
    merged.mkdir(exist_ok=True)
    titles=sorted(design['train'])
    directories=[root/(t+'-only') for t in titles]
    reports=[read(p/'report.json') for p in directories]
    parent_file=d/base['parent']/'camera-policy.pt'
    if digest(parent_file)!=design['parent']:
        raise ValueError('Common parent changed')
    for title,p,r in zip(titles,directories,reports):
        if (r['encoder']!=design['parent'] or r['sft']!=digest(p/'camera-policy.pt') or
            {s['sha256'] for s in r['train']}!=set(design['train'][title]) or r['steps']!=750):
            raise ValueError('Specialist data, update count or initialization differs')
    plan=dict(method='Equal arithmetic average of four common-parent SFT parameter tensors. No calibration, routing, extra gradient updates or held-based coefficient fitting.',
              parent=design['parent'],weights={r['sft']:1/4 for r in reports},
              evaluation='Compare the frozen merged actor with the shared four-title actor and each same-title specialist on the exact repeated development files. Offline human imitation only.')
    reg=merged/'protocol.json'
    if reg.is_file() and read(reg)!=plan:raise ValueError('Merge protocol differs')
    reg.write_text(json.dumps(plan,indent=2)+'\n')
    load=lambda f:torch.load(f,weights_only=True,map_location='cpu')
    parent=load(parent_file)
    states=[load(p/'camera-policy.pt') for p in directories]
    if any(s.keys()!=parent.keys() for s in states):raise ValueError('Parameter names differ')
    averaged={}
    for k,v in parent.items():
        tensors=[s[k] for s in states]
        if any(t.shape!=v.shape or t.dtype!=v.dtype for t in tensors):raise ValueError('Architecture mismatch')
        if v.is_floating_point():
            combined=torch.stack([t.double() for t in tensors]).mean(0)
            if not combined.isfinite().all():raise ValueError('Nonfinite weights')
            averaged[k]=combined.to(v.dtype)
        else:
            if any(not torch.equal(t,tensors[0]) for t in tensors):raise ValueError('Unequal discrete model state')
            averaged[k]=tensors[0]
    model=CameraPolicy().eval()
    model.load_state_dict(averaged)
    reference_file=merged/'camera-policy.pt'
    deployed_file=merged/'policy.ts'
    if not (merged/'report.json').exists():
        torch.save(averaged,reference_file)
        probe=torch.rand((4,2,112,96),generator=torch.Generator().manual_seed(261126))
        with torch.inference_mode():
            compiled=torch.jit.trace(model,probe[:1])
            compiled.save(str(deployed_file))
            exported=torch.jit.load(str(deployed_file))
            for got,expected in zip(exported(probe),model(probe)):
                torch.testing.assert_close(got,expected)
        (merged/'report.json').write_text(json.dumps(dict(encoder=design['parent'],sft=digest(reference_file),
                torchscript_sha256=digest(deployed_file),train=[s for r in reports for s in r['train']],
                additional_optimizer_updates=0,merged_from=plan['weights']),indent=2)+'\n')
    state_on_disk=load(reference_file)
    if any(not torch.equal(state_on_disk[k],v) for k,v in averaged.items()):raise ValueError('Merge artifact mismatch')
    split=read(d/base['base_split'])['split']['held']
    lookup={r['sha256']:d/r['file'] for r in split}
    lookup.update({r['archive_sha256']:d/r['path'] for r in read(d/'human-retail17/held-qualified-02-stages-frozen.json')['qualified']})
    for t,p in zip(titles,directories):
        files=[lookup[h] for h in design['held'][t]]
        if any(digest(f)!=h for f,h in zip(files,design['held'][t])):raise ValueError('Changed held photos')
        output=merged/'evaluation'/t
        for module,dest in [('compare_human_camera_policies',output),('score_human_bomb_onsets',output/'bomb-onsets.json')]:
            subprocess.run(list(map(str,[sys.executable,'-m','scripts.'+module,'--held',*files,
                                          '--models',joint,p,merged,'--out',dest])),check=True)
        print('Verified merged policy on',t,flush=True)


if __name__=='__main__':
    main()
