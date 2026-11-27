"""Export both registered camera feedback arms for the H1 original-game actors."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn

from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze
from scripts.train_replay_camera_encoder import CameraPolicy


class CameraWithDirection(nn.Module):
    def __init__(self, camera, residual, feedback):
        super().__init__()
        self.features = nn.Sequential(camera.spatial, camera.policy[:-1])
        self.head = camera.policy[-1]
        self.residual = residual
        self.feedback = feedback

    def forward(self, image, previous_direction):
        z = self.features(image)
        move,focus,shot,bomb = self.head(z).split((9,2,2,2),1)
        prior = previous_direction if self.feedback else torch.full_like(previous_direction,4)
        return self.residual(z,move,prior),focus,shot,bomb


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    d=p.parse_args().data.resolve()
    root=d/'four-title-direction-feedback'
    source=d/'specialist-vs-shared/merged-equal'
    plan=json.loads((root/'protocol.json').read_bytes())
    report=json.loads((root/'report.json').read_bytes())
    if not report['original_test_gate'] or digest(source/'policy.ts')!=plan['parent'] or digest(root/'protocol.json')!=report['registration']:
        raise ValueError('Missing qualified camera parent or feedback registration')
    state=torch.load(source/'camera-policy.pt',map_location='cpu',weights_only=True)
    camera=CameraPolicy()
    if any('policy.spatial_context.' in k for k in state):
        camera.add_spatial_context()
    camera.load_state_dict(state)
    camera.eval()
    parent=torch.jit.load(str(source/'policy.ts'),map_location='cpu').eval()
    torch.set_num_threads(2)
    torch.manual_seed(1511017)
    fixture=torch.rand(9,2,112,96)
    all_moves=torch.arange(9,dtype=torch.long)
    with torch.inference_mode():
        frozen=parent(fixture)
        for arm in ('image','feedback'):
            residual_file=root/(arm+'.ts')
            if digest(residual_file)!=report['model_sha256'][arm]:
                raise ValueError('Changed residual checkpoint')
            residual=torch.jit.load(str(residual_file),map_location='cpu').eval()
            model=CameraWithDirection(camera,residual,arm=='feedback').eval()
            destination=root/arm
            destination.mkdir(exist_ok=True)
            file=destination/'policy.ts'
            if file.exists():
                deployed=torch.jit.load(str(file),map_location='cpu').eval()
            else:
                deployed=torch.jit.trace(model,(fixture[:1],all_moves[:1]))
            z=camera.policy[:-1](camera.spatial(fixture))
            expected=(residual(z,frozen[0],all_moves if arm=='feedback' else torch.full_like(all_moves,4)),*frozen[1:4])
            actual=deployed(fixture,all_moves)
            for a,b in zip(actual,expected,strict=True):
                torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
            if not file.exists():
                tmp=file.with_name('staged-policy.ts')
                deployed.save(str(tmp))
                tmp.replace(file)
            freeze(destination/'export.json',dict(policy_sha256=digest(file),residual_sha256=digest(residual_file),
                parent_sha256=plan['parent'],observation='112x96 grayscale and causal absolute difference',
                previous_direction='4 on reset, then previous sampled direction in [0,8]',tested_previous_values=list(range(9)),
                learned_heads=['direction'],other_heads='Unmodified parent focus, shoot and cast logits'))
            print(arm,digest(file),flush=True)


if __name__=='__main__':
    main()
