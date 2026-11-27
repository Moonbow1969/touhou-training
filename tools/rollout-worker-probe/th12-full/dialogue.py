"""Offline native TH12 Normal Reimu A dialogue scoring on 112 by 96 actor frames."""
import numpy as np


def portrait_features(frames):
    """Unit gradient vectors of the fixed left portrait and dialogue-box region."""
    crop=np.asarray(frames,dtype=np.float32)[:,37:102,1:49]
    gx=crop[:,1:-1,2:]-crop[:,1:-1,:-2]
    gy=crop[:,2:,1:-1]-crop[:,:-2,1:-1]
    descriptor=np.stack((gx,gy),axis=-1).reshape(len(crop),-1)
    np.clip(descriptor,-120,120,out=descriptor)
    descriptor/=np.maximum(np.linalg.norm(descriptor,axis=1,keepdims=True),1e-6)
    return descriptor


def score_frames(frames, prototypes, chunk=128):
    scores=[]
    for start in range(0,len(frames),chunk):
        x=portrait_features(frames[start:start+chunk])
        scores.append((x@prototypes.T).max(axis=1))
    return np.concatenate(scores) if scores else np.empty(0,dtype=np.float32)


def dialogue_mask(scores, threshold, negative_max):
    """Require an adjacent agreeing decision before removing survival reward."""
    above=scores>=threshold
    confirmed=np.zeros(len(scores),dtype=bool)
    adjacent=above[1:] & above[:-1]
    confirmed[:-1]|=adjacent
    confirmed[1:]|=adjacent
    uncertain=(scores>negative_max) & ~confirmed
    return confirmed,uncertain


def survival_overlay(reward_parts, confirmed):
    if reward_parts.ndim!=2 or reward_parts.shape!=(len(confirmed),5):
        raise ValueError("Expected five native TH12 reward components per decision.")
    overlay=reward_parts.copy()
    overlay[confirmed,0]=0
    return overlay


if __name__=="__main__":
    import argparse
    import hashlib
    p=argparse.ArgumentParser(description="Write a separate TH12 Nazrin-dialogue survival overlay.")
    p.add_argument("source",help="Signed TH12 camera trajectory NPZ")
    p.add_argument("model",help="Frozen dialogue prototype NPZ")
    p.add_argument("output",help="New output NPZ; existing files are refused")
    args=p.parse_args()
    from pathlib import Path
    with np.load(args.model) as m,np.load(args.source) as trajectory:
        frames=trajectory["frames"]
        if frames.ndim!=3 or frames.shape[1:]!=(112,96) or frames.dtype!=np.uint8:
            raise ValueError("Expected original 112 by 96 uint8 grayscale pre-action frames.")
        scores=score_frames(frames,m["prototypes"])
        mask,uncertain=dialogue_mask(scores,m["threshold"],m["train_negative_max"])
        parts=survival_overlay(trajectory["reward_parts"],mask)
    with Path(args.output).open("xb") as f:
        np.savez_compressed(f,source_sha256=hashlib.sha256(Path(args.source).read_bytes()).hexdigest(),
                            model_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
                            dialogue=mask,uncertain=uncertain,score=scores,
                            reward_parts=parts,reward=parts.sum(axis=1))
