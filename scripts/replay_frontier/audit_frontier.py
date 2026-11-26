"""Inventory train-only TH14 originals, validate motor reconstruction and screen Stage 2 openings offline."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
if not (REPO / "scripts/train_demo_route.py").is_file():
    REPO = REPO / "work/th14-rl"
ROOT = REPO.parents[1]
sys.path.insert(0, str(REPO))
from replays.inspect_rpy import read_gameplay_payload, u32
from replays.build_corpus import player_group, split_for
from scripts.train_demo_route import RouteTracker, SCALE, features

D = ROOT / "work/th14-human-demos"
O = ROOT / "outputs/touhou-sft-world"
F = O / "sft-native-data-261123/frozen-training-snapshot"
A = O / "th14-demo-alice-route-261125"
OUT = ROOT / "outputs/th14-training-replay-frontier-261125"
IDS = ("0613", "05e0", "068f")
ALTS = ("093f", "0591", "04fb")
SWAP_END = 496
JOIN = 656
LIMIT = 720


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def controls(words):
    # Native four-tick displacements in simultaneous opposite inputs identify left/down priority.
    x = np.where(words & 64, -1, (words & 128) > 0)
    y = np.where(words & 32, 1, -((words & 16) > 0).astype(int))
    return 9 * ((words & 8) > 0) + 3 * (1 + y) + 1 + x


def integrate(words, origin, delta, bounds):
    pos = np.empty((len(words) + 1, 2))
    pos[0] = origin
    for t, displacement in enumerate(delta[controls(words)]):
        pos[t + 1] = np.minimum(np.maximum(pos[t] + displacement, bounds[0]), bounds[1])
    return pos


def statistics(values):
    return dict(n=len(values), median=float(np.median(values)), p95=float(np.quantile(values, .95)),
                max=float(np.max(values)), within_one=float(np.mean(values <= 1.)))


def read_original(entry):
    path = D / entry['name']
    assert sha(path) == entry['sha256']
    raw = path.read_bytes()
    ptr = u32(raw, 12)
    user = raw[ptr + 12:ptr + 400].decode('cp932', errors='replace')
    assert 'Stage All Clear' in user and 'Normal' in user and '1.00b' in user
    b = read_gameplay_payload(path)
    offset = 0x94
    stocks, streams = [], []
    with np.load(path.with_name(path.stem + '-actions.npz'), allow_pickle=False) as cache:
        for s in range(1, 7):
            assert b[offset] == s
            n, length = u32(b, offset + 4), u32(b, offset + 8)
            assert length == n * 6 + (n + 29) // 30
            keys = np.frombuffer(b, dtype='<u2', count=3*n, offset=offset+0xdc).reshape(-1,3)[:,0].copy()
            assert np.array_equal(keys,cache[f'stage_{s}_input_words'])
            streams.append(keys)
            stocks.append(dict(stage=s, frames=n, lives=b[offset+0x50], life_pieces=b[offset+0x54],
                               bombs=b[offset+0x5c], bomb_pieces=b[offset+0x60], power=u32(b,offset+0x44),
                               graze=u32(b,offset+0x2c), bomb_edges=int(np.count_nonzero(
                                   np.diff(np.r_[0,((keys & 2)>0).astype(int)])==1))))
            offset += length+0xdc
    assert offset == len(b)
    return stocks, streams


def original_key(route, tick):
    f, keys = route['frame_2'], route['keys_2']
    idx = np.clip(np.searchsorted(f,tick,side='right')-1,0,len(f)-1)
    return keys[idx, np.clip(tick-f[idx],0,3)]


def original_position(route, tick):
    return np.stack([np.interp(tick,route['frame_2'],v) for v in route['positions_2'].T],axis=-1)


def make_opening(route, predicted, words):
    # Keep original Alice actions after the demonstration window. Return its spatial offset to zero smoothly.
    t = np.arange(JOIN+5)
    position = original_position(route,t)
    fade = np.clip((t-SWAP_END)/(JOIN-SWAP_END),0,1)
    position += (1-fade*fade*(3-2*fade))[:,None] * (predicted[SWAP_END]-position[SWAP_END])
    position[:SWAP_END+1] = predicted[:SWAP_END+1]
    keys = original_key(route,t).copy()
    keys[:SWAP_END] = words[:SWAP_END]
    return np.clip(position,*route['bounds']).astype(np.float32), keys


def evaluate_field(path, route, model, openings):
    with np.load(path,allow_pickle=False) as tape:
        col = {key:i for i,key in enumerate(json.loads(str(tape['metadata_json']))['native_field_order'])}
        b,a=tape['native_before'],tape['native_after']
        idx=np.flatnonzero((b[:,col['stage']]==2)&(b[:,col['stage_frame']]<=LIMIT))
        if not len(idx):
            return None
        f=b[idx,col['stage_frame']]
        resets=np.flatnonzero(np.diff(f)<0)+1
        if len(resets):
            idx=idx[resets[-1]:]
        elif f[0]>32:
            return None
        idx=idx[(a[idx,col['stage']]==2)&(a[idx,col['stage_frame']]>b[idx,col['stage_frame']])]
        bullets=tape['bullets'][idx]
        original=tape['positions_before'][idx,:2]
        frame=b[idx,col['stage_frame']]
        hit=(b[idx,col['player_state']]==1)&(a[idx,col['player_state']]==4)
        first=int(frame[hit][0]) if np.any(hit) else None
        minimum_margin=100.
        result={}
        for name, obj in openings.items():
            sim=original[0].copy()
            minima=[]; fields_present=[]; errors=[]; hit_margins=[]
            for i,row in enumerate(idx):
                t=int(frame[i]); ticks=int(a[row,col['stage_frame']]-t)
                if ticks>8:
                    raise ValueError('Unexpected native frame interval')
                if obj is None or t>=JOIN:
                    target=original_position(route,t+4)
                    key=original_key(route,t)
                else:
                    pos,keys=obj;target=pos[t+4];key=keys[t]
                product=features(sim,target,np.asarray(bool(key&8)),np.asarray(bool(key&1)),route['bounds'])
                with torch.inference_mode():
                    arm=int(model(torch.from_numpy(product))[0].argmax())
                delta=route['action_deltas'][arm]/4
                future=np.clip(sim[None]+np.arange(ticks+1)[:,None]*delta,*route['bounds'])
                bullet=bullets[i]
                bullet=bullet[(bullet[:,5]==1)&(bullet[:,6]==1)&(bullet[:,4]>0)]
                margin=minimum_margin
                if len(bullet):
                    world=original[i]+bullet[:,:2]*[384,448]
                    upcoming=world[None]+np.arange(1,ticks+1)[:,None,None]*bullet[None,:,2:4]*16
                    # Check both ends of player/projectile update ordering on the frozen field.
                    margin=min(np.min(np.linalg.norm(upcoming-future[:-1,None],axis=-1)-bullet[:,4]*32-1),
                               np.min(np.linalg.norm(upcoming-future[1:,None],axis=-1)-bullet[:,4]*32-1))
                minima.append(float(margin))
                fields_present.append(bool(len(bullet)))
                errors.append(float(np.linalg.norm(sim-original_position(route,t))))
                if hit[i]:
                    hit_margins.append(dict(frame=t,margin=float(margin)))
                sim=future[-1]
            m=np.asarray(minima);pre=frame<=(first if first is not None else SWAP_END)
            stay=frame<=SWAP_END
            result[name]=dict(prefix_contact_intervals=int(np.sum(m[pre]<=0)),
                prefix_within_8=int(np.sum(m[pre]<8)), prefix_min_margin=float(np.min(m[pre])),
                at_original_onset=hit_margins, through_496_contacts=int(np.sum(m[stay]<=0)),
                bridge_contacts=int(np.sum(m[(frame>SWAP_END)&(frame<=JOIN)]<=0)),
                bridge_rows=int(np.sum((frame>SWAP_END)&(frame<=JOIN))),complete_bridge=bool(frame[-1]>=JOIN),
                bridge_within_8=int(np.sum(m[(frame>SWAP_END)&(frame<=JOIN)]<8)),
                return_error_at_656=(float(np.asarray(errors)[np.abs(frame-JOIN).argmin()]) if frame[-1]>=JOIN else None))
        return dict(tape_sha256=sha(path),n=len(idx),first_frame=int(frame[0]),last_frame=int(frame[-1]),
                    original_first_onset=first,field_rows=int(sum(fields_present)),alternatives=result)


def audit():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=D/'expanded-corpus.json';rows=json.loads(manifest.read_text(encoding='utf-8'))
    manifest_names={r['name'] for r in rows}
    extra=[]
    for path in sorted(D.glob('th14_*.rpy')):
        if path.name in manifest_names:
            continue
        with path.open('rb') as reader:
            head=reader.read(20)
            reader.seek(u32(head,12)+12)
            fields=dict(line.split(' ',1) for line in reader.read(370).decode('cp932',errors='replace').splitlines() if ' ' in line)
        if (fields.get('Chara','').strip().replace(' ','')!='ReimuA' or fields.get('Rank','').strip()!='Normal'
                or fields.get('Version','').strip()!='1.00b' or fields.get('Stage','').strip()!='All Clear'):
            continue
        name=fields['Name'].strip()
        if split_for(name)!='train':
            continue
        binary=read_gameplay_payload(path);cursor=0x94;counts={}
        for _ in range(binary[0x78]):
            stage=binary[cursor];n=u32(binary,cursor+4);length=u32(binary,cursor+8)
            assert length==6*n+(n+29)//30
            counts[stage]=n;cursor+=length+0xdc
        assert set(counts)==set(range(1,7)) and cursor==len(binary)
        extra.append(dict(name=path.name,player_group=player_group(name),sha256=sha(path),frames=counts,
                          split='train by frozen player grouping',ever_used_for_selection=False))
    rows=[r for r in rows if r['title']==14 and r['difficulty']=='Normal' and r['version']=='1.00b'
          and r['shot'].replace(' ','')=='ReimuA' and r['split']=='train' and set(r['frames_by_stage'])==set('123456')]
    with np.load(A/'route.npz',allow_pickle=False) as z:
        route={key:z[key] for key in z.files}
    model=RouteTracker().eval()
    model.load_state_dict(torch.load(A/'model.pt',weights_only=True,map_location='cpu'))
    source={}; reconstruction={}; openings={'alice':None}; raw={}
    for row in rows:
        name=Path(row['name']).stem; rid=name[-4:]
        headers,streams=read_original(row)
        paired=sorted(int(f.stem.rsplit('stage',1)[-1]) for f in F.glob(f'{name}-stage*.npz'))
        source[name]=dict(player_group=row['player_group'],original_sha256=row['sha256'],comment=row['comment'],
                          source=row['source'],paired_stages=paired,header_resources=headers)
        if rid not in IDS+ALTS:
            continue
        origin=np.array([0.,400.]);reconstruction[rid]={}
        for stage,words in enumerate(streams,1):
            pred=integrate(words,origin,route['action_deltas']/4,route['bounds']);origin=pred[-1]
            if stage==2 and rid in ALTS:
                openings[rid]=make_opening(route,pred,words)
                raw[rid]=(pred[:501].astype(np.float32),words[:501])
            if rid in IDS:
                src=F/f'{name}-stage{stage}.npz'
                with np.load(src,allow_pickle=False) as z:
                    f=z['stage_frame'];actual=z['context'][:,:2]*SCALE;stocks=z['context'][:,3]
                    err=np.linalg.norm(pred[f]-actual,axis=1)
                    local=integrate(words[f[0]:],actual[0],route['action_deltas']/4,route['bounds'])
                    anchored=np.linalg.norm(local[f-f[0]]-actual,axis=1)
                    reconstruction[rid][stage]=dict(shard=sha(src),opening=statistics(err[f<=500]),
                        full=statistics(err),opening_first_native_anchor=statistics(anchored[f<=500]),
                        observed_stock_losses=int(np.count_nonzero(np.diff(stocks)<0)),
                        input_offset=json.loads(src.with_suffix('.json').read_text())['offset'])
                    if rid=='05e0' and stage==2:
                        native=np.column_stack([np.interp(np.arange(501),f,v) for v in actual.T])
                        # The shard begins at frame 30. Fill only its unmeasured first 30 ticks by reconstruction.
                        native[:f[0]]=pred[:f[0]]
                        openings['05e0']=make_opening(route,native,words)
    # Commit the split and deterministic ranking rule before opening either secondary credit.
    fields={}
    for s in ('2201','2202','2402','2601','3301'):
        paths=list(O.glob(f'*{s}*/transition-shards/*.npz'))
        if len(paths)==1:
            payload=evaluate_field(paths[0],route,model,openings)
            if payload is not None:
                fields[s]=payload
    def order(name):
        dev=[fields[s]['alternatives'][name] for s in ('2201','2202')]
        return (sum(x['prefix_contact_intervals'] for x in dev),sum(x['prefix_within_8'] for x in dev),
                sum(x['bridge_contacts'] for x in dev),-min(x['prefix_min_margin'] for x in dev),name)
    ordered=sorted([k for k in openings if k!='alice'],key=order)
    candidates={}
    for k in ordered:
        pos,keys=openings[k]; target=OUT/f'{k}-opening.npz'
        values=dict(target_positions=pos,preferred_keys=keys)
        if k in raw:
            values.update(inferred_positions=raw[k][0],decoded_keys=raw[k][1])
        np.savez_compressed(target,**values)
        candidates[k]=dict(sha256=sha(target),rank_inputs=order(k)[:-1],source='aligned paired after frame 30' if k=='05e0' else 'inferred train replay positions',origin_xy=pos[0].tolist())
    report=dict(schema='th14-train-only-opening-frontier-v1',manifest_sha256=sha(manifest),train_clear_originals=len(rows),
        unique_train_player_groups=len({x['player_group'] for x in rows}),explicit_no_miss_ids=['0613',*ALTS],
        extra_local_unadmitted_originals=extra,all_local_unique_groups=len({x['player_group'] for x in rows+extra}),
        motor_fit_identity='0613', independent_paired_train_diagnostics=['05e0','068f'], source=source,
        train_motion_table=(route['action_deltas']/4).tolist(),native_bounds=route['bounds'].tolist(),
        coach_model_sha256=sha(A/'model.pt'),base_route_sha256=sha(A/'route.npz'),reconstruction=reconstruction,
        swap_until=SWAP_END,return_by=JOIN,selection_credits=['2201','2202'],available_check_credits=[k for k in fields if k not in ('2201','2202')],
        fixed_field_order=ordered,opening_candidates=candidates,field_proxies=fields,
        selection_rule='lexicographic pre-first-real-onset simulated contacts, <8px rows, bridge contacts, worst credit minimum, identifier. Checks never enter ranking.')
    (OUT/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(ordered)
    for s,v in fields.items():
        print(s,v['original_first_onset'],{key:(x['prefix_contact_intervals'],x['prefix_within_8'],x['bridge_contacts'],
            [(h['frame'],round(h['margin'],1)) for h in x['at_original_onset']],x['return_error_at_656'])
            for key,x in v['alternatives'].items()})


if __name__=='__main__':
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    audit()
