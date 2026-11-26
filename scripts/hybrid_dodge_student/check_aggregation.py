"""Exercise the prospective-label archive contract using only synthetic rows."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

from .actor import ROOT,REPO
from .aggregate import eligible
from .evaluate import PINNED,WEIGHT
from .ppo import ROLE
from scripts.checkpoint_storage import digest
from scripts.native_story_transitions import NATIVE_FIELDS


def main():
    with tempfile.TemporaryDirectory(prefix='synthetic-th14-aggregation-',dir=ROOT/'work') as tmp:
        work=Path(tmp).resolve()
        if not work.is_relative_to((ROOT/'work').resolve()):
            raise ValueError('Synthetic fixture escaped workspace.')
        trial=dict(block=1,order=1,arm='development_joint_ppo',policy_seed=7101)
        policy=dict(role=ROLE,source_weights_sha256=PINNED[WEIGHT])
        meta=dict(trial=trial,task=dict(difficulty='Normal',shot='Reimu A'),actor=policy,native_field_order=NATIVE_FIELDS)
        n=16
        before=np.zeros((n,len(NATIVE_FIELDS)),np.int64)
        for key,value in [('stage',1),('player_state',1),('lives',2),('bombs',3),('spell_id',-1),
                          ('global_tick',4*np.arange(n)),('stage_frame',30+4*np.arange(n))]:
            before[:,NATIVE_FIELDS.index(key)]=value
        after=before.copy()
        for key in ('global_tick','stage_frame'):after[:,NATIVE_FIELDS.index(key)]+=4
        ctx=np.zeros((n,10),np.float32);ctx[:,1]=400/448
        tape=work/'synthetic.npz';buffer=work/'synthetic.pt'
        np.savez(tape,metadata_json=json.dumps(meta),bullets=np.zeros((n,96,7),np.float32),context=ctx,
                 native_before=before,native_after=after,action=np.tile([4,0,1,0],(n,1)),
                 four_tick_stage_transition=np.ones(n,bool))
        torch.save(dict(role=ROLE,synthetic=True),buffer)
        episode=dict(trial=trial,result=dict(native_continues=0,transition_shard=dict(path=tape.name,sha256=digest(tape))),
                     onpolicy_buffer=dict(file=buffer.name,sha256=digest(buffer)))
        report=dict(status='complete',specification=dict(development=dict(role=ROLE),schedule=[trial],
                    policies={'development_joint_ppo':policy}),episodes=[episode])
        report_path=work/'report.json';report_path.write_text(json.dumps(report),encoding='utf-8')
        subprocess.run([sys.executable,'-m','scripts.hybrid_dodge_student.aggregate','--development-report',
                       str(report_path),'--out',str(work/'derived')],check=True,cwd=REPO)
        derived=json.loads((work/'derived/manifest.json').read_text(encoding='utf-8'))
        if derived['status']!='complete' or derived['credits'][0]['rows']!=n:
            raise AssertionError('Synthetic archive did not round-trip.')
        excluded=[]
        for reserved in (3601,4001):
            trial['policy_seed']=reserved
            try:eligible(report)
            except ValueError:excluded.append(reserved)
        if excluded!=[3601,4001]:
            raise AssertionError('Protected evaluation seeds passed prospective admission.')
        print(json.dumps(dict(synthetic_rows=n,rejected_evaluation_seeds=excluded)))


if __name__=='__main__':
    main()
