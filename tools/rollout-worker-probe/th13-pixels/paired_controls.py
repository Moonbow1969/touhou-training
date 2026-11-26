"""Compare chronological action agreement in the two common-controller-seed heldout pairs."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('out',type=Path)
    args=p.parse_args()
    out=[]
    for r in (7,8):
        pair=[]
        for lane in 'ab':
            folder=args.run/f'pg-{r:02d}-{lane}'
            obj=json.loads((folder/'credit.json').read_text());label=json.loads((folder/'label.json').read_text())
            d=np.load(folder/'behavior.npz');pair.append((obj,label,d['action']))
        (left,ll,la),(right,rl,ra)=pair
        assert left['seed']==right['seed'] and left['weights_sha256']!=right['weights_sha256']
        n=min(len(la),len(ra))
        out.append(dict(round=r,controller_seed=left['seed'],native_rng_controlled=False,common_decision_indices=n,
                        action_differences=int((la[:n]!=ra[:n]).sum()),
                        a=dict(model=left['model'],sha256=left['weights_sha256'],reward=ll['reward'],note=ll['note'],total_actions=len(la)),
                        b=dict(model=right['model'],sha256=right['weights_sha256'],reward=rl['reward'],note=rl['note'],total_actions=len(ra))))
    result=dict(title='TH13 Normal Reimu official 0.01a trial',metric='60s wall horizon stock; pending automatic death trance independently resolved from follow-up images',unit='independent original credit; common controller draws within each pair; game RNG uncontrolled',matched_pairs=out)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(out))
if __name__=='__main__':main()
