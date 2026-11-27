"""Independently check stock TH17 0.01a rank, stock, stage entry and physical evidence."""
import csv
import json

from scripts.checkpoint_storage import digest

BUILD='3ffa3963003cce6dc02b1dae5adeb4411138772834efdb195a05b9081027aaba'
OBSERVER='a561ed8f077b9d75fc380777722a67e59f9dbbad1dcbc1108c1083be8f1dff69'
OBSERVER_V2='803154279225a61f80ed1b6a8b7a1b8a094b94bfdabd2f08052c549f7898b2ee'


def next_tick_held_endpoint(folder,role):
    """Authenticate the first native death directly adjacent to a final live action observation."""
    if role!='held' or (folder/'worker.rc').read_text().strip()!='9' or (folder/'sensor.rc').read_text().strip()!='0':
        raise ValueError('Expected a scored held credit and the original worker/sensor records')
    pins=json.loads((folder/'deployed-source.json').read_bytes())
    if (pins['exe_sha256'],pins['observer_sha256'],pins['actor_sha256'])!=(BUILD,OBSERVER_V2,'95b44b132f7b2d91eb38c566c46dc89a5fb876295fd8db60a346aa436baa7fc8'):
        raise ValueError('Unqualified build for adjacent native death adjudication')
    with (folder/'decisions.tsv').open() as file: steps=list(csv.DictReader(file,delimiter='\t'))
    with (folder/'native.tsv').open() as file: trace=[{k:int(v) for k,v in r.items()} for r in csv.DictReader(file,delimiter='\t') if int(r['stage'])>=1]
    final={k:int(steps[-1][k]) for k in ('action_t','onset_ns','observed_ns','move','focus','shoot','bomb','keys','observed_keys','stage','after_stage','tick','after_tick','spares','bombs')}
    if (len(steps)<2 or int(steps[-1]['action_t'])!=len(steps)-1 or any(int(x['action_t'])!=i or x['keys']!=x['observed_keys'] for i,x in enumerate(steps))
            or final['onset_ns']>=final['observed_ns'] or final['stage']!=final['after_stage'] or final['spares']!=0):
        raise ValueError('Final measured policy action not a qualified, live original command')
    die=next((i for i,x in enumerate(trace) if x['spares']==-1),None)
    if die is None or die<1 or any(x['spares']!=-1 for x in trace[die:]):
        raise ValueError('Original native final life was not exhausted irreversibly')
    live,dead=trace[die-1:die+1]
    if (live['stage'],live['tick'],live['spares'],live['bombs'])!=(final['after_stage'],final['after_tick'],0,final['bombs']) or (dead['stage'],dead['tick'])!=(live['stage'],live['tick']+1) or dead['ms']<live['ms']:
        raise ValueError('Missing or non-adjacent measured action and original death')
    return dict(rule='native-next-tick-death-held-v1',role=role,optimizer_eligible=False,
                controllable_original_ticks_between_observations=0,final_action=final,last_live_native=live,
                first_dead_native=dead,postdeath_rows=len(trace)-die,
                raw_hashes={k:digest(folder/k) for k in ('decisions.tsv','native.tsv','deployed-source.json','terminal.png','worker.rc','sensor.rc')})


def verify_stage_chronology(native):
    """The original stage timer can restart at a new stage; music can restart at any time."""
    for row in native:
        if row['stage'] not in (1,2,3) or row['tick']<0:
            raise ValueError('Invalid TH17 native stage or stage tick')
    for prior,current in zip(native,native[1:]):
        transition=current['stage']-prior['stage']
        if transition not in (0,1) or transition==0 and current['tick']<prior['tick'] or current['ms']<prior['ms']:
            raise ValueError('TH17 stage sequence, per-stage clock or acquisition clock went backwards')


def inspect(receipt_file,r,expected):
    folder=receipt_file.parent
    source={1:OBSERVER,2:OBSERVER_V2}.get(r.get('observer_version',1))
    if source is None or r['source_executable_sha256']!=BUILD or r['native_observer_sha256']!=source or r['difficulty']!=expected:
        raise ValueError('Unexpected TH17 build, observer or difficulty')
    for name,field in (('native.tsv','native_trace_sha256'),('decisions.tsv','decisions_sha256'),
                       ('observations.u8','frames_sha256'),('rank-initial.png','rank_initial_sha256'),
                       ('rank-selected.png','rank_selected_sha256'),('start-hud.png','start_hud_sha256'),
                       ('terminal.png','terminal_observation_sha256')):
        if digest(folder/name)!=r[field]:raise ValueError(f'Changed TH17 {name}')
    with (folder/'native.tsv').open(encoding='utf8') as stream:
        native=[{k:int(v) for k,v in row.items()} for row in csv.DictReader(stream,delimiter='\t') if int(row['stage'])>=1]
    leading = 0
    if r.get('observer_version',1)==2:
        while leading<len(native):
            birth=native[leading]
            if (birth['stage'],birth['start_stage'],birth['tick'],birth['power'],birth['spares'],birth['bombs'],birth['continues'],birth['score'])!=(1,1,0,0,0,0,0,0):
                break
            leading+=1
    if leading!=r.get('native_preinitialization_rows',0):raise ValueError('Unexpected pre-play stock initialization')
    native=native[leading:]
    if len(native)!=r['difficulty_verified_native_samples'] or not native:raise ValueError('Incomplete TH17 original samples')
    first,last=native[0],native[-1]
    if tuple(first[k] for k in ('stage','start_stage','spares','bombs'))!=(1,1,2,3) or not 0<=first['tick']<180:
        raise ValueError('Missing TH17 original full-stock opening')
    if any((row['rank'],row['continues'],row['character'],row['beast'])!=(expected,0,0,0) for row in native):
        raise ValueError('TH17 native difficulty, character, beast or Continue changed')
    if r['terminal_screen'] not in ('game_over','score_ranking') or last['spares']!=-1:
        raise ValueError('TH17 terminal requires independent qualification')
    if r.get('adjacent_terminal_evidence_sha256'):
        f=folder/'held-terminal-boundary.json'
        if digest(f)!=r['adjacent_terminal_evidence_sha256'] or json.loads(f.read_bytes())!=next_tick_held_endpoint(folder,r['split']):
            raise ValueError('Adjacent held terminal evidence failed native replay')
    verify_stage_chronology(native)
    entering=next((x for x in native if x['stage']==2),None)
    entry=(bool(entering),entering['spares'] if entering else None,entering['bombs'] if entering else None)
    live=next(x for x in reversed(native) if x['spares']>=0)
    if entry!=(r['stage2_reached'],r['stage2_spare_lives'],r['stage2_bombs']) or (live['bombs'],last['bombs'])!=(r['last_live_bombs'],r['remaining_bombs']):
        raise ValueError('TH17 original stage or stock ledger differs')
    casts=sum(max(0,x['bombs']-y['bombs']) for x,y in zip(native,native[1:]) if x['spares']==y['spares'])
    return dict(difficulty=expected,difficulty_verified_native_samples=len(native),difficulty_evidence_sha256=r['native_trace_sha256'],
                stage2_reached=entry[0],stage2_spares=entry[1],stage2_bombs=entry[2],effective_bombs=casts,
                stage2_evidence_sha256=r['native_trace_sha256'],terminal_bombs=last['bombs'],last_playable_bombs=live['bombs'],
                last_playable_stage1_tick=max(row['tick'] for row in native if row['stage']==1 and row['spares']>=0))


