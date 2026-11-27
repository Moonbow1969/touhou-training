"""Control for averaging's smaller parameter displacement from the common parent."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch
from scripts.checkpoint_storage import digest
from scripts.train_replay_camera_encoder import CameraPolicy


def read(f):
    return json.loads(f.read_bytes())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    d=parser.parse_args().data.resolve()
    root=d/'specialist-vs-shared'
    joint=d/'ranked-multigame-sft/four-title'
    design=read(d/'ranked-multigame-sft/design.json')
    parent=d/design['parent']
    merged=root/'merged-equal'
    control=root/'joint-matched-distance'
    control.mkdir(exist_ok=True)
    def load(p):
        r=read(p/'report.json')
        f=p/'camera-policy.pt'
        if digest(f)!=r['sft']:raise ValueError('Frozen model mismatch')
        return torch.load(f,weights_only=True,map_location='cpu')
    base,shared,average=map(load,(parent,joint,merged))
    names=[n for n,v in base.items() if v.is_floating_point() and n.startswith(('spatial.','policy.'))]
    if not names:raise ValueError('Cannot identify deployed actor parameters')
    def distance(model):
        return sum((model[n].double()-base[n].double()).square().sum() for n in names).sqrt().item()
    m,j=distance(average),distance(shared)
    alpha=m/j
    plan=dict(method='Match actor distance from the common 38-pilot parent by scaling the joint update. Compute the scalar from deployed spatial and policy parameter tensors only, before scoring. Held observations do not determine the coefficient.',
              parent_sha=digest(parent/'policy.ts'),joint_sha=digest(joint/'policy.ts'),merge_sha=digest(merged/'policy.ts'),
              merged_distance=m,joint_distance=j,alpha=alpha,scope=names)
    registration=control/'protocol.json'
    if registration.exists() and read(registration)!=plan:raise ValueError('Distance control registration differs')
    registration.write_text(json.dumps(plan,indent=2)+'\n')
    state={n:(v.double()+alpha*(shared[n].double()-v.double())).to(v.dtype) if v.is_floating_point() else v
           for n,v in base.items()}
    if abs(distance(state)-m)>1e-5*m:raise ValueError('Distance validation failed')
    if not (control/'report.json').exists():
        model=CameraPolicy().eval()
        model.load_state_dict(state)
        torch.save(state,control/'camera-policy.pt')
        probe=torch.rand(4,2,112,96,generator=torch.Generator().manual_seed(261127))
        traced=torch.jit.trace(model,probe[:1])
        traced.save(str(control/'policy.ts'))
        for got,expected in zip(torch.jit.load(str(control/'policy.ts'))(probe),model(probe)):
            torch.testing.assert_close(got,expected)
        (control/'report.json').write_text(json.dumps(dict(train=read(joint/'report.json')['train'],
            sft=digest(control/'camera-policy.pt'),torchscript_sha256=digest(control/'policy.ts'),
            additional_optimizer_updates=0),indent=2)+'\n')
    files={r['sha256']:d/r['file'] for r in read(d/design['base_split'])['split']['held']}
    files.update({r['archive_sha256']:d/r['path'] for r in read(d/'human-retail17/held-qualified-02-stages-frozen.json')['qualified']})
    for title,hashes in read(root/'protocol.json')['held'].items():
        paths=[files[h] for h in hashes]
        if any(digest(f)!=h for f,h in zip(paths,hashes)):raise ValueError('Changed held stage')
        subprocess.run(list(map(str,[sys.executable,'-m','scripts.compare_human_camera_policies','--models',
                                     parent,joint,control,merged,'--held',*paths,'--out',control/'evaluation'/title])),check=True)
    print('Verified shrinkage control',alpha,flush=True)


if __name__=='__main__':
    main()
