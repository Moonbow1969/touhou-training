"""Compose a frozen state student and a frozen learned twelve-tick contact head."""
import json
from pathlib import Path

import torch

from scripts.checkpoint_storage import digest
from scripts.relative_phase_student import StockRelativeActor
from scripts.train_anticipatory_contact import AnticipatoryContact
from scripts.train_local_avoidance import local_context

STUDENT_HASH="d5bfabb005ee68f2bcc2410e6649f9df528262d28c3f215a35b2c38fbc3acdde"
CONTACT_HASH="7700eb95c641a1c95bdf9c4a7d51d4137a17755fe70ab673e99fff8a3301d124"


def load_risk(path):
    if digest(path)!=CONTACT_HASH:raise ValueError("Unexpected contact head.")
    payload=torch.load(path,weights_only=True,map_location="cpu")
    if payload["schema"]!="th14-learned-contact-12tick-v1":raise ValueError("Unexpected risk horizon.")
    model=AnticipatoryContact().eval();model.load_state_dict(payload["state_dict"])
    return model


def adjust(logits,risk,weight):
    idx=torch.arange(len(logits),device=logits.device)
    base=logits.argmax(1);focus=base//9;move=base%9
    fixed=logits.reshape(-1,2,9)[idx,focus];cost=risk.reshape(-1,2,9)[idx,focus]
    # Outside model-predicted contact, the two-head policy returns the original action exactly.
    enabled=cost[idx,move]>=.10
    new=(fixed-weight*cost).argmax(1)
    return torch.where(enabled,new,move),focus


class ContactStudent(StockRelativeActor):
    def __init__(self,student,risk,configuration):
        super().__init__(student);config=json.loads(Path(configuration).read_bytes())
        if digest(student)!=STUDENT_HASH or not config["development_gate"] or config["contact_hash"]!=CONTACT_HASH:
            raise ValueError("Unexpected calibration or weights.")
        self.contact=load_risk(risk);self.weight=config["chosen_weight"];self.last=None
        self.model.register_forward_hook(self.capture)

    def capture(self,module,inputs,outputs):self.last=outputs[0]

    @torch.inference_mode()
    def predict(self,observation,deterministic=False):
        self.last=None;action,_=super().predict(observation,deterministic)
        if self.environment.snapshot.player_state!=1 or action[3] or "x" in self.environment.held:return action,None
        b=torch.from_numpy(observation["bullets"][None]);c=torch.from_numpy(observation["context"][None])
        p=self.contact(b,local_context(c,0)).sigmoid()[:,:18]
        move,focus=adjust(self.last,p,self.weight)
        if focus.item()!=int(action[1]):raise RuntimeError("Frozen focus changed.")
        action[0]=move.item()
        return action,None
