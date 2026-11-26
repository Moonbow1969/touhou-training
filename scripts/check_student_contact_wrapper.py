"""Exercise candidate inference row-by-row on a complete held-out native credit."""
from types import SimpleNamespace
import json
import numpy as np
import torch

from scripts.audit_student_contact import C,RISK,STUDENT,ROOT,WORLD
from scripts.student_contact_policy import ContactStudent


def main():
    torch.set_num_threads(1)
    out=ROOT/"outputs/th14-relative-student-contact-261125"
    policy=ContactStudent(STUDENT,RISK,out/"calibration.json")
    env=SimpleNamespace(reset_count=1);policy.environment=env
    shard=next((WORLD/"th14-relative-student-ending-2601-261125/transition-shards").glob("*.npz"))
    with np.load(shard) as z:
        src=z["native_before"];ctx=z["context"];bul=z["bullets"];base=z["action"]
    changed=others=0
    for i,r in enumerate(src):
        if not r[C["snapshot_valid"]]:continue
        hp=int(r[C["boss_hp"]]);sp=int(r[C["spell_id"]]);size=int(r[C["boss_max_hp"]])
        env.last_frame=int(r[C["stage_frame"]]);env.held={"x"} if r[C["previous_x"]] else set()
        env.game=SimpleNamespace(read_game_val=lambda key:int(r[C["bombs"]]))
        env.snapshot=SimpleNamespace(stage=int(r[C["stage"]]),player_state=int(r[C["player_state"]]),
            spell_id=None if sp<0 else sp,boss=None if size<0 else SimpleNamespace(max_hp=size,hp=hp,vulnerable=r[C["boss_vulnerable"]]==1))
        act,_=policy.predict(dict(context=ctx[i],bullets=bul[i]))
        changed+=int(act[0]!=base[i,0]);others+=int(np.any(act[1:]!=base[i,1:]))
    expected=json.loads((out/"check-2601.json").read_bytes())["all"]["changed"]
    if changed!=expected or others:raise RuntimeError(f"Native wrapper mismatch: changed {changed}, expected {expected}, other heads {others}.")
    res=dict(seed=2601,rows=len(src),movement_changed=changed,any_other_head_changed=others,batch_expected=expected,
             mode="archived native observations replayed through deployment class without opening any game")
    (out/"wrapper-check.json").write_text(json.dumps(res,indent=2));print(json.dumps(res))


if __name__=="__main__":main()
