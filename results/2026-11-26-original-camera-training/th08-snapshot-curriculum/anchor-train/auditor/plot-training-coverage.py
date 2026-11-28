"""Plot the actual original-game Stage 1 training exposure of a mixed-source manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def render(manifest,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    record=json.loads(manifest.read_bytes());groups={'ordinary':[],'anchor':[]};spans={'ordinary':[],'anchor':[]}
    for source in record['rows']:
        kind=source['source_kind'];p=manifest.parent/source['file']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==source['sha256']
        with np.load(p,allow_pickle=False) as t:
            tick=t['native_tick'][t['native_stage']==1].astype(int)
            if len(tick):groups[kind].append(tick);spans[kind].append([int(tick.min()),int(tick.max()),len(tick)])
    tallies={}
    for k,rows in groups.items():
        values=np.concatenate(rows)
        tallies[k]=dict(qualified_games_or_fragments=len(rows),stage1_decisions=len(values),min_native_tick=int(values.min()),max_native_tick=int(values.max()),
            median_native_tick=float(np.median(values)),decisions_from_tick2201=int((values>=2201).sum()),decisions_from_tick3000=int((values>=3000).sum()),
            fraction_from_tick2201=float((values>=2201).mean()),fraction_from_tick3000=float((values>=3000).mean()))
    fig,(ax,span)=plt.subplots(1,2,figsize=(10.5,4.5),gridspec_kw={'width_ratios':[1.25,1]})
    colors={'ordinary':'#246491','anchor':'#bb5a35'}
    names={'ordinary':'Original Game Start','anchor':'Restored at tick 2,201'}
    allmax=max(v['max_native_tick']for v in tallies.values());bins=np.arange(0,allmax+251,250)
    for k,rows in groups.items():
        values=np.concatenate(rows);counts,_=np.histogram(values,bins=bins)
        ax.stairs(counts/len(values),bins,color=colors[k],linewidth=2,label=names[k])
        ax.stairs(counts/len(values),bins,color=colors[k],alpha=.1,fill=True)
    ax.axvline(2201,color='#686868',linestyle=':',linewidth=.8);ax.set_xlabel('Native Stage 1 tick');ax.set_ylabel('Share of source decisions per 250 ticks');ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.legend(frameon=False,fontsize=8);ax.set_title('Where the policy acted',loc='left',fontsize=11)
    pos=0;lab=[];ticks=[]
    for key in ('ordinary','anchor'):
        intervals=sorted(spans[key],key=lambda r:r[1]);start=pos
        for lo,hi,_ in intervals:
            span.hlines(pos,lo,hi,linewidth=1.3,color=colors[key],alpha=.8);pos+=1
        ticks.append((start+pos-1)/2);lab.append('Ordinary' if key=='ordinary'else 'Restored');pos+=3
    span.set_yticks(ticks,labels=lab);span.invert_yaxis();span.set_xlim(ax.get_xlim());span.set_xlabel('Native Stage 1 tick');span.set_title('One interval per admitted source',loc='left',fontsize=11)
    for axis in (ax,span):
        axis.spines[['top','right']].set_visible(False);axis.grid(axis='x',alpha=.15,linewidth=.6);axis.tick_params(labelsize=8)
    fig.suptitle('TH08 Normal Ghost training exposure',x=.08,ha='left',fontsize=15,fontweight='bold')
    fig.text(.08,.018,'Ordinary credits run to Game Over. Restored fragments stop at the first spare loss or after 3,000 native ticks.',fontsize=8,color='#454545')
    fig.tight_layout(rect=(0,.055,1,.94));out.mkdir(parents=True,exist_ok=True);fig.savefig(out/'source-native-exposure.png',dpi=170,bbox_inches='tight');plt.close(fig)
    (out/'source-native-exposure.json').write_text(json.dumps(dict(sources=tallies,optimizer_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()),indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('manifest',type=Path);parser.add_argument('out',type=Path);args=parser.parse_args();render(args.manifest,args.out)
