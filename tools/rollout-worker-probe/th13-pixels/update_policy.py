"""One credit-balanced on-policy REINFORCE update, scored entirely by observed native HUD."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from policy_credit import features


def categorical(F,W):
    logits=F@W.T
    p=np.exp(logits-logits.max(1,keepdims=True))
    return p/p.sum(1,keepdims=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial',type=Path,required=True);p.add_argument('--credit',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    W=np.load(args.initial)['weights']
    fingerprint=hashlib.sha256(W.astype('<f8').tobytes()).hexdigest()
    returns=[];scores=[];feature_sets=[];before=[];episodes=[]
    for folder in args.credit:
        report=json.loads((folder/'credit.json').read_text()); label=json.loads((folder/'label.json').read_text())
        if label['status'] not in ['active','resolved_trance','game_over'] or label['reward'] is None:
            continue
        assert fingerprint==report['weights_sha256']
        d=np.load(folder/'behavior.npz');F=features(d['obs'][:-1]);a=d['action'];probs=categorical(F,W)
        assert np.allclose(probs,d['prob'],rtol=2e-9,atol=2e-9)
        score=(np.eye(3)[a]-probs).T@F
        returns.append(float(label['reward']));scores.append(score);feature_sets.append(F);before.append(probs)
        episodes.append(folder.name)
    if len(scores)<3:raise ValueError('At least three independently classified matching-policy credits are required.')
    r=np.asarray(returns); baseline=(r.sum()-r)/(len(r)-1);adv=r-baseline
    g=np.mean(adv[:,None,None]*np.asarray(scores),axis=0)
    # An ascent update on the credit return, clipped in Euclidean norm, with an empirical KL bound.
    rate=.18;norm=np.linalg.norm(g)
    if norm>2:g*=2/norm
    def mean_kl(candidate):
        return float(np.mean([np.sum(p*(np.log(p)-np.log(categorical(f,candidate))),axis=1).mean() for f,p in zip(feature_sets,before)]))
    candidate=W+rate*g
    while mean_kl(candidate)>.01:
        rate*=.5;candidate=W+rate*g
    np.savez(args.output,weights=candidate)
    result=dict(estimator='mean over credits of leave-one-other-credit-out advantage times sum of categorical score functions',episodes=episodes,returns=returns,baseline=baseline.tolist(),gradient_norm_before_clip=float(norm),gradient_clip=2.0,learning_rate=rate,empirical_forward_kl=mean_kl(candidate),moved=bool(np.any(candidate!=W)),frozen_parent_sha256=fingerprint,weights_sha256=hashlib.sha256(candidate.astype('<f8').tobytes()).hexdigest(),feature_contract='bias, 42 current mean 16x16 bins, 42 abs pixel temporal difference means; 112x96 grayscale scaled /255; three logits in left,neutral,right order')
    args.output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
