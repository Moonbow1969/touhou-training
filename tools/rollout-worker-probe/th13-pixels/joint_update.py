"""One synchronous, title-balanced policy gradient update from original TH12 and TH13 actors."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from policy_credit import features
from update_policy import categorical


def raw_sha(weights):
    return hashlib.sha256(weights.astype("<f8").tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--initial",type=Path,required=True)
    p.add_argument("--credits",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    weights=np.load(args.initial,allow_pickle=False)["weights"]
    if weights.shape!=(3,85):
        raise ValueError("Expected the shared left/neutral/right pixel actor.")
    initial_hash=raw_sha(weights)
    manifest=json.loads(args.credits.read_bytes())
    gradients={}
    records=[]
    seen=set()
    batch={}
    for title in (12,13):
        entries=[r for r in manifest if r["title"]==title]
        if len(entries)!=4 or len({e["episode"] for e in entries})!=4:
            raise ValueError("Each title needs its four preassigned training credits.")
        scores=[]
        trials=[]
        returns=[]
        for row in entries:
            path=args.credits.parent/row["behavior"]
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen or digest!=row["sha256"] or row["actor_sha256"]!=initial_hash:
                raise ValueError("Duplicate data, changed inputs or stale behavior policy.")
            seen.add(digest)
            with np.load(path,allow_pickle=False) as z:
                action=z["action"].astype(int)
                x=features(z["obs"][:-1])
                prob=categorical(x,weights)
                if not np.array_equal(action,action.clip(0,2)) or len(action)!=len(x) or len(x)==0:
                    raise ValueError("Invalid action stream.")
                if not np.allclose(prob,z["prob"],atol=2e-9,rtol=2e-9):
                    raise ValueError("Executed behavior probabilities do not match the common weights.")
                if not np.allclose(np.log(prob[np.arange(len(action)),action]),z["logp"],atol=2e-8,rtol=2e-8):
                    raise ValueError("Logged action likelihoods do not replay.")
            reward=float(row["scaled_return"])
            if not np.isfinite(reward) or abs(reward)>1.000001:
                raise ValueError("Out-of-range per-title normalized return.")
            returns.append(reward)
            scores.append((np.eye(3)[action]-prob).T@x)
            trials.append((x,prob))
            records.append({k:v for k,v in row.items() if k!="behavior"}|dict(decisions=len(action)))
        ret=np.asarray(returns)
        base=(ret.sum()-ret)/(len(ret)-1)
        gradients[title]=np.mean((ret-base)[:,None,None]*np.stack(scores),axis=0)
        batch[title]=trials
    gradient=(gradients[12]+gradients[13])/2
    norm=float(np.linalg.norm(gradient))
    if norm<1e-12:
        raise ValueError("The observed returns produced zero joint policy gradient.")
    delta=gradient*min(1.,2/norm)
    def divergences(w):
        return {t:float(np.mean([np.mean(np.sum(p*(np.log(p)-np.log(categorical(x,w))),axis=1))
                                 for x,p in batch[t]])) for t in batch}
    rate=.18
    while rate>1e-8 and max(divergences(weights+rate*delta).values())>.01:
        rate/=2
    updated=weights+rate*delta
    args.out.mkdir(exist_ok=False)
    np.savez(args.out/"joint.npz",weights=updated)
    norms={t:float(np.linalg.norm(v)) for t,v in gradients.items()}
    cosine=float(np.sum(gradients[12]*gradients[13])/(norms[12]*norms[13])) if min(norms.values())>0 else None
    result=dict(algorithm="On-policy episodic REINFORCE, leave-one-other-credit-out baseline within each title, equal weighted mean of two title gradients.",
                parent_sha256=initial_hash,weights_sha256=raw_sha(updated),credits=records,per_title_gradient_norm=norms,
                per_title_gradient_cosine=cosine,joint_unclipped_gradient_norm=norm,step_size=rate,
                per_title_empirical_forward_kl=divergences(updated),held_performance="Separate fresh native comparison required.")
    (args.out/"update.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
