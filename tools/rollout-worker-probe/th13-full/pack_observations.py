"""Serialize exact actor images and passive TH13 labels for a complete original training credit."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

label_dtype = np.dtype([('t0','<i8'),('t1','<i8'),('pre','<i4',15),('post','<i4',15)])

def stable_labels(rows):
    pre, post = rows['pre'], rows['post']
    xy = np.ascontiguousarray(pre[:,6:8]).view('<f4')
    xy_after = np.ascontiguousarray(post[:,6:8]).view('<f4')
    bounds = np.ascontiguousarray(pre[:,8:12]).view('<f4')
    center = (bounds[:,:2] + bounds[:,2:]) / 2
    durations = (rows['t1'] - rows['t0']) / 1e6
    stable = ((pre[:,0]>0)&(post[:,0]>0)&(pre[:,0]==pre[:,14])&(post[:,0]==post[:,14])&
              (pre[:,2]==post[:,2])&(pre[:,3]==post[:,3])&(pre[:,4]==post[:,4])&
              (pre[:,13]==2)&(post[:,13]==2)&(np.max(np.abs(xy-xy_after),axis=1)<.01)&
              (np.max(np.abs(xy-center),axis=1)<.03)&(np.abs(xy[:,0])<=192)&
              (xy[:,1]>=16)&(xy[:,1]<=448)&(durations>0)&(durations<18))
    return xy, xy_after, bounds, stable, durations

def pack(images, labels, rollout, out):
    n=len(images)
    required=(n,112,96)
    if images.shape!=required or len(labels)!=n or rollout['features'].shape!=(n,110):
        raise ValueError('Exact one observation per sampled step is required')
    if rollout['terminated'].sum()!=1 or not rollout['terminated'][-1]:
        raise ValueError('Require exactly one genuine terminal')
    xy,xy_after,bounds,stable,durations=stable_labels(labels)
    # Check raw timing and alignment against the actual policy input on all decisions.
    pooled=images.reshape(n,7,16,6,16).mean(axis=(2,4)).reshape(n,42) / 255.0
    max_abs_pool=float(np.max(np.abs(pooled-rollout['features'][:,:42])))
    max_abs_motion=0.
    for k in range(1,n,96):
        end=min(n,k+96)
        delta=np.abs(images[k:end].astype(np.int16)-images[k-1:end-1].astype(np.int16))
        pooled_delta=delta.reshape(end-k,7,16,6,16).mean(axis=(2,4)).reshape(end-k,42)/255.
        max_abs_motion=max(max_abs_motion,float(np.max(np.abs(pooled_delta-rollout['features'][k:end,42:84]))))
    if max(max_abs_pool,max_abs_motion)>1e-6 or rollout['features'][0,42:84].any():
        raise ValueError('Raw observation cannot reproduce frozen policy-time features')
    np.savez_compressed(out, obs=images,xy=xy,xy_after=xy_after,bounds=bounds,stable=stable,
                        action=rollout['action'],stage=labels['pre'][:,3],spares=labels['pre'][:,4],
                        bombs=labels['pre'][:,5],native_input=labels['pre'][:,12],
                        native_frame_before=labels['pre'][:,2],native_frame_after=labels['post'][:,2],
                        start_ns=labels['t0'],finish_ns=labels['t1'],policy_sha256=rollout['policy_sha256'])
    return dict(steps=n,stable=int(stable.sum()),capture_p95_ms=float(np.percentile(durations,95)),
                brightness_recomputed_max_abs=max_abs_pool,motion_recomputed_max_abs=max_abs_motion,
                observation='Uninterrupted exact pre-action 112x96 uint8 capture, one image per actor step; passive original labels only.',
                sha256=hashlib.sha256(out.read_bytes()).hexdigest())

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path);p.add_argument('rollout',type=Path);p.add_argument('out',type=Path)
    args=p.parse_args()
    images=np.fromfile(args.folder/'observations.u8','u1').reshape(-1,112,96)
    rows=np.fromfile(args.folder/'observation-labels.bin',label_dtype)
    with np.load(args.rollout,allow_pickle=False) as z: print(json.dumps(pack(images,rows,z,args.out),indent=2))
