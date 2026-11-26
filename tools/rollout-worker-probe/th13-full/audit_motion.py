"""Compare sampled movement to the next original player position on complete training credits."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np

def measure(source):
    out=defaultdict(list)
    for path in sorted(source.glob('train-*-*.npz')):
        with np.load(path) as z:
            xy=z['xy'];delta=np.diff(xy,axis=0)
            valid=z['stable'][:-1]&z['stable'][1:]&(z['stage'][:-1]==z['stage'][1:])&(z['spares'][:-1]==z['spares'][1:])
            ticks=np.diff(z['native_frame_before'])
            valid &= (ticks>=1)&(ticks<=10)&(np.abs(delta).max(axis=1)<45)
            pre=xy[:-1];a=z['action'][:-1,0]
            valid &= (np.abs(pre[:,0])<=184)&(pre[:,1]>=32)&(pre[:,1]<=432)
            move=np.stack((a%3-1,a//3-1),axis=1)
            for key,val in [('x',pre[:,0]),('y',pre[:,1]),('dx',delta[:,0]),('dy',delta[:,1]),('horizontal',move[:,0]),('vertical',move[:,1])]:
                out[key].append(val[valid])
    if not out:raise ValueError('No exact native training credits')
    vals={k:np.concatenate(v) for k,v in out.items()}
    x,y,dx,dy,mx,my=(vals[k] for k in ('x','y','dx','dy','horizontal','vertical'))
    wall=( ((mx<0)&(x<=-183.99))|((mx>0)&(x>=183.99))|((my<0)&(y<=32.01))|((my>0)&(y>=431.99)) )
    motion=(mx!=0)|(my!=0)
    result=dict(credits=len(out['x']),valid_adjacent_decisions=len(y),movement_requests=int(motion.sum()),
                nonneutral_with_at_least_one_outward_axis=int((motion&wall).sum()),actual_displacement_over_one_pixel=int((motion&(np.maximum(abs(dx),abs(dy))>1)).sum()),
                at_bottom=int((y>=431.99).sum()),down_requests=int((my>0).sum()),
                downward_requests_at_bottom=int(((my>0)&(y>=431.99)).sum()),directions={},
                source_files=sorted(p.name for p in source.glob('train-*-*.npz')),
                quality='Adjacent, stable original player labels with same stage and life count; 1–10 positive game ticks apart; absolute motion below 45 pixels; only the playable rectangle. Movement uses original between-decision displacement. The policy never reads these labels.',
                measured_clamp='x=+/-184, y=32 or 432 on this original build; conservative axis-interior x in (-160,160), y in (50,410).')
    for name,command,measured,interior,sign in [('left',mx,dx,abs(x)<160,-1),('right',mx,dx,abs(x)<160,1),('up',my,dy,(y>50)&(y<410),-1),('down',my,dy,(y>50)&(y<410),1)]:
        wanted=(command==sign)
        result['directions'][name]={}
        for pop,mask in [('all',wanted),('interior',wanted&interior)]:
            v=measured[mask]
            result['directions'][name][pop]=dict(requested=int(len(v)),moved_in_requested_direction_over_one_pixel=int((v*sign>1).sum()),median_displacement=float(np.median(v)) if len(v) else None)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path);args=p.parse_args()
    args.output.write_text(json.dumps(measure(args.source),indent=2)+'\n')
