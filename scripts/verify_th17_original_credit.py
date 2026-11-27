"""Independently check stock TH17 0.01a rank, stock, stage entry and physical evidence."""
import csv

from scripts.checkpoint_storage import digest

BUILD='3ffa3963003cce6dc02b1dae5adeb4411138772834efdb195a05b9081027aaba'
OBSERVER='a561ed8f077b9d75fc380777722a67e59f9dbbad1dcbc1108c1083be8f1dff69'


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
    if r['source_executable_sha256']!=BUILD or r['native_observer_sha256']!=OBSERVER or r['difficulty']!=expected:
        raise ValueError('Unexpected TH17 build, observer or difficulty')
    for name,field in (('native.tsv','native_trace_sha256'),('decisions.tsv','decisions_sha256'),
                       ('observations.u8','frames_sha256'),('rank-initial.png','rank_initial_sha256'),
                       ('rank-selected.png','rank_selected_sha256'),('start-hud.png','start_hud_sha256'),
                       ('terminal.png','terminal_observation_sha256')):
        if digest(folder/name)!=r[field]:raise ValueError(f'Changed TH17 {name}')
    with (folder/'native.tsv').open(encoding='utf8') as stream:
        native=[{k:int(v) for k,v in row.items()} for row in csv.DictReader(stream,delimiter='\t') if int(row['stage'])>=1]
    if len(native)!=r['difficulty_verified_native_samples'] or not native:raise ValueError('Incomplete TH17 original samples')
    first,last=native[0],native[-1]
    if tuple(first[k] for k in ('stage','start_stage','spares','bombs'))!=(1,1,2,3) or not 0<=first['tick']<180:
        raise ValueError('Missing TH17 original full-stock opening')
    if any((row['rank'],row['continues'],row['character'],row['beast'])!=(expected,0,0,0) for row in native):
        raise ValueError('TH17 native difficulty, character, beast or Continue changed')
    if r['terminal_screen'] not in ('game_over','score_ranking') or last['spares']!=-1:
        raise ValueError('TH17 terminal requires independent qualification')
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
