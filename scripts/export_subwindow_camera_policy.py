"""Package the frozen visual encoder and both four-control subwindow predictions for native collectors."""
import argparse
import json
from pathlib import Path
import torch
from torch import nn
from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze
from scripts.train_four_title_subwindow_sft import SubwindowHead
from scripts.train_portable_full_sft import WIDTHS
from scripts.train_replay_camera_encoder import CameraPolicy

class PlannedControls(nn.Module):
    def __init__(self,camera,head,repeated):
        super().__init__()
        self.camera=camera
        self.head=head
        self.repeated=repeated

    def forward(self,images):
        h=self.camera.policy[:-1](self.camera.spatial(images))
        predictions=self.head(h,self.camera.policy[-1](h))
        if self.repeated:predictions=predictions.mean(1,keepdim=True).expand(-1,2,-1)
        return predictions.split(WIDTHS,dim=2)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    d=parser.parse_args().data.resolve();root=d/'four-title-subwindow-sft'
    plan=json.loads((root/'protocol.json').read_bytes());result=json.loads((root/'report.json').read_bytes())
    parent=d/'ranked-multigame-sft/four-title-expanded-twelve/camera-policy.pt'
    if digest(parent)!=plan['parent'] or result['registered']!=digest(root/'protocol.json'):raise ValueError('Unexpected parent')
    torch.set_num_threads(1)
    camera=CameraPolicy()
    state=torch.load(parent,map_location='cpu',weights_only=True)
    if any('policy.spatial_context.' in k for k in state):camera.add_spatial_context()
    camera.load_state_dict(state)
    sample=torch.rand((4,2,112,96),generator=torch.Generator().manual_seed(271699))
    artifacts={}
    for name,checksum in result['weights'].items():
        fitted=root/f'{name}.pt'
        if digest(fitted)!=checksum:raise ValueError('Changed supervised residual')
        head=SubwindowHead();head.load_state_dict(torch.load(fitted,map_location='cpu',weights_only=True))
        model=PlannedControls(camera,head,name=='four-tick').eval()
        with torch.inference_mode():
            actor=torch.jit.trace(model,sample[:1])
            dst=root/f'{name}-policy.ts'
            if not dst.exists():actor.save(str(dst))
            restored=torch.jit.load(str(dst),map_location='cpu')
            errors=[]
            for expected,actual,width in zip(model(sample),restored(sample),WIDTHS,strict=True):
                if actual.shape!=(4,2,width):raise ValueError('Missing original control head')
                torch.testing.assert_close(actual,expected)
                errors.append(float((expected-actual).abs().max()))
        artifacts[name]=dict(sha256=digest(dst),source_residual=checksum,max_head_errors=errors)
    freeze(root/'export.json',dict(registration=digest(root/'protocol.json'),models=artifacts,
        interface='One current plus previous-difference observation, shape [batch,2,112,96], yields four logit tensors [batch,2,num_actions]. The four-tick control samples slot zero once and holds the same joint command for all four game ticks. The two-subwindow candidate samples both slots from their respective logits and executes two ticks each. No second image or mid-hold inference. These models expose policy logits only.'))
if __name__=='__main__':
    main()
