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
    occupancy,x_edges,y_edges=np.histogram2d(x,y,bins=(np.linspace(-192,192,25),np.linspace(0,448,29)))
    result=dict(credits=len(out['x']),valid_adjacent_decisions=len(y),movement_requests=int(motion.sum()),
                nonneutral_with_at_least_one_outward_axis=int((motion&wall).sum()),actual_displacement_over_one_pixel=int((motion&(np.maximum(abs(dx),abs(dy))>1)).sum()),
                at_bottom=int((y>=431.99).sum()),down_requests=int((my>0).sum()),
                downward_requests_at_bottom=int(((my>0)&(y>=431.99)).sum()),directions={},
                occupancy={'x_edges':x_edges.tolist(),'y_edges':y_edges.tolist(),'counts_y_x':occupancy.T.astype(int).tolist()},
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

def plot(result,destination):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig,axes=plt.subplots(1,2,figsize=(9.6,5.1),gridspec_kw={'width_ratios':[1,1.1]})
    occupancy=result['occupancy']
    heat=axes[0].pcolormesh(occupancy['x_edges'],occupancy['y_edges'],np.ma.masked_equal(occupancy['counts_y_x'],0),
                            cmap='inferno',norm=LogNorm(vmin=1))
    axes[0].set_ylim(448,0);axes[0].set_xlim(-192,192);axes[0].set_aspect('equal')
    axes[0].set_xlabel('Native horizontal coordinate');axes[0].set_ylabel('Native vertical coordinate')
    axes[0].set_title('Original player positions')
    fig.colorbar(heat,ax=axes[0],fraction=.04,pad=.03,label='Decisions per spatial cell')
    x=np.arange(4)
    for pos,label,color,offset in [('all','All valid positions','#536dae',-.17),('interior','Interior on the movement axis','#008773',.17)]:
        rates=[]
        for key in ('left','right','up','down'):
            r=result['directions'][key][pos];rates.append(r['moved_in_requested_direction_over_one_pixel']/r['requested'])
        axes[1].bar(x+offset,rates,width=.31,color=color,label=label)
        for i,v in enumerate(rates):axes[1].text(i+offset,v+.017,f'{v:.0%}',ha='center',fontsize=8)
    axes[1].set_xticks(x,['Left','Right','Up','Down']);axes[1].set_ylim(0,1.11)
    axes[1].set_ylabel('Requested movement producing >1 native pixel');axes[1].set_title('Observed movement by requested axis')
    axes[1].spines[['top','right']].set_visible(False);axes[1].grid(axis='y',alpha=.17)
    axes[1].legend(loc='lower left',fontsize=8)
    fig.suptitle('TH13 PPO2  ·  Native position readback across eight training credits',fontsize=14)
    fig.text(.5,.015,f"{result['downward_requests_at_bottom']:,} of {result['down_requests']:,} down requests came at the lower boundary. The actor received pixels only.",ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.94));fig.savefig(destination,dpi=155);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--figure',type=Path);args=p.parse_args()
    result=measure(args.source)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    if args.figure:plot(result,args.figure)
