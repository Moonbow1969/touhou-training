"""Measure Bomb inventory before each stock loss and original TH17 fragment progress."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from scripts.checkpoint_storage import digest
from scripts.score_original_camera_run import records
from scripts.train_five_original_games import freeze


def audit(path,receipt,title):
    if title==17:
        trace=path.parent/'native.tsv';expected=receipt['native_trace_sha256'];skip=receipt.get('native_preinitialization_rows',0)
    else:
        trace=path.parent.parent/'raw'/receipt['episode']/'engineering-native.tsv'
        expected=receipt['covered_native_events']['raw_sha256']['engineering-native.tsv'];skip=0
    if digest(trace)!=expected:raise ValueError('Changed native inventory record')
    losses=[];pieces=[];prior=None
    with trace.open(encoding='utf8') as stream:
        for row in csv.DictReader(stream,delimiter='\t'):
            if int(row['stage'])<1:continue
            if title==12 and (int(row['mode'])!=0 or row['stage']!=row['manager_stage'] or int(row['tick'])<0):continue
            if skip:
                if int(row['tick'])!=0 or int(row['spares'])!=0 or int(row['bombs'])!=0:raise ValueError('Unexpected TH17 pre-game row')
                skip-=1;continue
            fields=['stage','tick','spares','bombs']+(['life_pieces','bomb_pieces'] if title==17 else [])
            current={k:int(row[k]) for k in fields}
            if prior is None:
                if (current['stage'],current['spares'],current['bombs'])!=(1,2,2 if title==12 else 3):raise ValueError('Unexpected starting stock')
            else:
                change=current['spares']-prior['spares']
                if change<0:
                    if change!=-1:raise ValueError('Ambiguous native stock loss')
                    losses.append(dict(stage=current['stage'],tick=current['tick'],previous_tick=prior['tick'],
                        lost_active_or_spare_life=prior['spares'],bombs_before_loss=prior['bombs'],bombs_at_loss=current['bombs']))
                if title==17 and change==0 and current['stage']==prior['stage']:
                    for key in ('life_pieces','bomb_pieces'):
                        if current[key]>prior[key]:
                            pieces.append(dict(field=key,before=prior[key],after=current[key],tick=current['tick']))
            prior=current
            if current['spares']==-1:break
    if not losses or prior['spares']!=-1:raise ValueError('Missing final exhaustion')
    return dict(seed=receipt['seed'],rank=receipt['difficulty'],native_sha256=expected,original_receipt=digest(path),
        losses=losses,same_life_fragment_increases=pieces if title==17 else None)


def main():
    arg=argparse.ArgumentParser(description=__doc__);arg.add_argument('--data',type=Path,required=True)
    arg.add_argument('--study',required=True,choices=('five-game-generation3','six-game-return-horizon'))
    a=arg.parse_args();d=a.data.resolve();s=d/a.study;root=d/'original-resource-loss-audit';root.mkdir(exist_ok=True)
    plan=json.loads((s/'camera-protocol.json').read_bytes())
    required={(t,k) for t in (12,17) for k in plan['train_seeds'][str(t)]};seen={};cells={}
    freeze(root/(s.name+'-protocol.json'),dict(source=digest(s/'camera-protocol.json'),
        design='Read every original native active-life decrement through the first full-stock exhaustion in the TH12 and TH17 training allocation. Record Bomb inventory from the immediately preceding observed sample and its native tick. For TH17 also count only positive fragment-counter changes at constant life stock within the same stage, reporting both events and games. Conversion on a stock or stage change is outside this measure. These are observed resource events. They do not tell us whether using a Bomb would have prevented a death or which item produced an increase.'))
    for t,f in records(s,True,True):
        if t not in (12,17):continue
        rec=json.loads(f.read_bytes());key=t,rec['seed']
        if key not in required:continue
        if key in seen or rec['policy_sha256']!=plan['behavior_sha256']:raise ValueError('Unexpected training credit')
        seen[key]=audit(f,rec,t)
    if set(seen)!=required:raise ValueError('Incomplete native resource population')
    for t in (12,17):
        for rank in (2,3):
            group=[v for (title,_),v in seen.items() if title==t and v['rank']==rank]
            if len(group)!=16:raise ValueError('Incomplete title/rank cell')
            b=[event['bombs_before_loss'] for z in group for event in z['losses']]
            c=dict(credits=len(group),losses=len(b),with_bombs=sum(x>0 for x in b),before_loss_stock_histogram=dict(Counter(map(str,b))))
            if t==17:
                c['fragment_increase_events']={key:sum(sum(e['field']==key for e in z['same_life_fragment_increases']) for z in group) for key in ('life_pieces','bomb_pieces')}
                c['credits_with_fragment_increase']={key:sum(any(e['field']==key for e in z['same_life_fragment_increases']) for z in group) for key in ('life_pieces','bomb_pieces')}
            cells[f'TH{t}-{rank}']=c
    freeze(root/(s.name+'.json'),dict(source=digest(s/'camera-protocol.json'),cells=cells,
                                credits={f'TH{t}-{s}':v for (t,s),v in sorted(seen.items())}))
    print(cells,flush=True)


if __name__=='__main__':
    main()
