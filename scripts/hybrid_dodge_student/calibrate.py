"""Freeze a conservative operating point using three training observers and the old 2402 validation credit."""
import json
import numpy as np
import torch

from .actor import OUT,MovementStudent,AnticipatoryContact
from .prepare import CONTACT,WORLD,BASE,dataset,SOURCE
from .train import evaluate
from scripts.relative_phase_student import RelativeActor
from scripts.evaluate_stage_policies import read_policy
from scripts.checkpoint_storage import digest


def main():
    torch.set_num_threads(2)
    if (OUT/'conservative.pt').exists():
        raise FileExistsError('The conservative candidate is already frozen in this artifact directory.')
    path=OUT/'student.pt'; weights=torch.load(path,weights_only=True,map_location='cpu')
    if digest(path)!='fb13c2f48d7501752b6cfa1d2417a61d3f8fd775d50de392a027828c6f5100d4':
        raise ValueError('The first generation changed.')
    net=MovementStudent().eval();net.load_state_dict(weights['movement'])
    contact=AnticipatoryContact().eval();contact.load_state_dict(weights['contact'])
    relative=RelativeActor().eval();relative.load_state_dict(weights['state_dict'])
    sft=read_policy(WORLD/'overnight-title-equal-61/sft/policy.pt','cpu',True)
    sets={};meta={}
    for name in ('late5','late6','practice4','2402'):
        sets[name],meta[name]=dataset(name,contact,relative,sft)
    grid=(-2.,-1.,0.,1.,2.,3.,4.,5.,6.,8.,10.)
    candidates=[]
    for bias in grid:
        net.keep_offset.fill_(bias)
        stats={name:evaluate(net,d)['all'] for name,d in sets.items()}
        quiet={name:s['quiet_changes']/max(1,s['quiet_rows']) for name,s in stats.items()}
        bc_drop={name:(s['prior_motion_agreement']-s['student_motion_agreement'])/s['measured_motion_rows']
               for name,s in stats.items() if s['measured_motion_rows']}
        allowed=max(quiet.values())<=.025 and max(bc_drop.values())<=.02
        candidates.append(dict(bias=bias,admitted=allowed,calibration_negative=stats['2402']['student_negative'],
                       calibration_anchor=stats['2402']['prior_negative'],quiet=quiet,bc_loss=bc_drop))
    winner=min((r for r in candidates if r['admitted']),key=lambda r:(r['calibration_negative'],-r['bias']))
    net.keep_offset.fill_(winner['bias'])
    weights['movement']=net.state_dict(); final=OUT/'conservative.pt'
    torch.save(weights,final)
    sha=digest(final)
    pre=dict(criteria='Calibration 2402 negative count lowest, larger keep logit breaks ties. All four group quiet changes at most 0.025; all three training groups motion agreement drop at most 0.02.',
        final_gate='Frozen-student 3601 Stage2 baseline negatives >=5, at least 20% fewer student negatives, quiet changes <=2.5%, actual-onset projected gains exceed losses.',
        selected=winner,grid=candidates,sha256=sha,v1_sha256=digest(path),calibration_sources=meta)
    (OUT/'freeze.json').write_text(json.dumps(pre,indent=2),encoding='utf-8')
    print('LOCKED',sha,winner,flush=True)
    for name in ('held4','held5','2601','3301','3601'):
        sets[name],meta[name]=dataset(name,contact,relative,sft)
    # This is the only credit reserved after the chosen design was fixed.
    final_name='frozen3601'
    trail=list(WORLD.glob('th14-relative-full-eval-frozen_student-3601*/transition-shards/*.npz'))
    if len(trail)!=1:
        raise FileNotFoundError('Final separate frozen-student 3601 credit has not yet produced exactly one transition shard.')
    SOURCE[final_name]=trail[0]
    sets[final_name],meta[final_name]=dataset(final_name,contact,relative,sft)
    report={name:evaluate(net,d) for name,d in sets.items()}
    s=report[final_name]['stage2']
    passes=(s['prior_negative']>=5 and s['student_negative']<=.8*s['prior_negative']
          and s['quiet_changes']<=.025*s['quiet_rows'] and s['onsets_projected_gain']>s['onsets_projected_loss'])
    assert digest(final)==sha
    result=dict(final_credit_gate=bool(passes),final_credit=final_name,weights_sha256=sha,
                sources=meta,metrics=report)
    (OUT/'frozen-checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for n in report:
        print(n,[(key,[(report[n][key][k]) for k in ('n','prior_negative','student_negative','quiet_changes','quiet_rows','onset_rows','onsets_projected_gain','onsets_projected_loss','prior_motion_agreement','student_motion_agreement')]) for key in ('all','stage2','stage2_first500') if key in report[n]])
    print('GATE',passes,flush=True)


if __name__=='__main__':
    main()
