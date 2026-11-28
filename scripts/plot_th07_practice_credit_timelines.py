from pathlib import Path
import json,csv,sys,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot(path):
 data=json.loads((path/'physical-outcomes.json').read_bytes())
 fig,ax=plt.subplots(figsize=(8.2,4.0),layout='constrained')
 for i,row in enumerate(data['rows']):
  b=path/'raw'/str(row['seed']);native=list(csv.DictReader((b/'native-input.csv').open()));p=list(csv.DictReader((b/'spell-phase.csv').open()))
  start=4; end=int(row['end_tick']);spell=[int(a['stage_tick']) for a in p if a['supervisor']=='2' and a['active']=='1' and a['spell_index']=='7'];earliest=min(spell) if spell else end
  ax.plot([start,earliest],[i,i],color='#7797ac',linewidth=5,solid_capstyle='butt')
  if earliest<end:ax.plot([earliest,end],[i,i],color='#4a365e',linewidth=5,solid_capstyle='butt')
  ticks=[int(native[j]['tick']) for j in range(1,len(native)) if int(native[j]['misses'])>int(native[j-1]['misses'])]
  ax.scatter(ticks,[i]*len(ticks),c='#fb9d40',marker='|',s=120,zorder=4)
  clear=row['physical_outcome']=='original_stage_clear'
  ax.scatter(end,i,facecolor=('white' if not row['behavior_qualified'] else '#226344' if clear else '#a84237'),edgecolor=('#226344' if clear else '#a84237'),marker='o' if clear else 'D',s=42,zorder=5)
 ax.invert_yaxis();ax.set_yticks(range(len(data['rows'])),[str(x['seed']) for x in data['rows']]);ax.set_xlabel('Original Stage 1 ticks since practice entry');ax.set_title('TH07 Normal · Reimu A · two spare lives, three Bombs, Power 128',loc='left',fontsize=12)
 ax.legend(handles=[Line2D([0],[0],lw=5,color='#7797ac',label='Before final card'),Line2D([0],[0],lw=5,color='#4a365e',label='From card ID 7'),Line2D([0],[0],ls='none',marker='|',color='#d57b1f',label='Spare loss'),Line2D([0],[0],ls='none',marker='o',color='#226344',label='Stage Clear'),Line2D([0],[0],ls='none',marker='D',color='#a84237',label='Game Over'),Line2D([0],[0],ls='none',marker='o',markerfacecolor='white',color='#555',label='Timing exclusion')],ncols=3,frameon=False,fontsize=8,loc='upper left',bbox_to_anchor=(0,-.18))
 ax.grid(axis='x',color='#e6e6e6',lw=.7);ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
 fig.savefig(path/'native-credit-timelines.png',dpi=180);plt.close(fig)
if __name__=='__main__':plot(Path(sys.argv[1]))
