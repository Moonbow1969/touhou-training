"""Partition native TH17 wall time using per-credit calibrated clocks, container identities and event marks."""
import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

def stats(values):
    return dict(n=len(values),mean_s=statistics.mean(values),median_s=statistics.median(values),
                minimum_s=min(values),maximum_s=max(values))

COMPONENTS=[
    ('game_worker_termination','reset_started','previous_wine_game_process_exited'),
    ('game_exec_dispatch','previous_wine_game_process_exited','new_wine_game_process_dispatched'),
    ('game_configuration_dialog_wait','new_wine_game_process_dispatched','original_config_dialog_seen'),
    ('game_configuration_physical_selection','original_config_dialog_seen','original_config_submitted'),
    ('client_window_wait_and_focus','original_config_submitted','original_client_mapped_and_focused'),
    ('client_title_ready','original_client_mapped_and_focused','original_title_ready'),
    ('open_rank_menu','original_title_ready','opened_original_rank_menu'),
    ('photograph_initial_rank','opened_original_rank_menu','default_rank_photo_written'),
    ('select_and_photograph_registered_rank','default_rank_photo_written','selected_rank_photo_verified'),
    ('open_character_menu','selected_rank_photo_verified','reimu_verified'),
    ('open_beast_menu','reimu_verified','wolf_verified'),
    ('prepare_original_observer','wolf_verified','native_sensor_shared_state_ready'),
    ('launch_learned_actor','native_sensor_shared_state_ready','learned_worker_dispatched'),
    ('first_native_action_after_actor_launch','learned_worker_dispatched','first_action'),
    ('observe_terminal_actor_exit','last_action','completed_worker_detected'),
    ('capture_terminal_screen','completed_worker_detected','original_terminal_photo_written'),
    ('transfer_native_tsv','original_terminal_photo_written','transferred_native.tsv'),
    ('transfer_uncompressed_frames','transferred_native.tsv','transferred_observations.u8'),
    ('transfer_actions','transferred_observations.u8','transferred_decisions.tsv'),
    ('compress_hash_and_seal','transferred_decisions.tsv','compressed_hashed_and_sealed'),
    ('read_and_hash_model','scheduler_entered_credit','checkpoint_read_and_sha256'),
    ('verify_deployed_executable_actor_and_observer','checkpoint_read_and_sha256','deployed_native_binaries_verified'),
    ('cross_clock_and_container_measurement','deployed_native_binaries_verified','reset_started'),
]

def read(path):
    a=json.loads(path.read_bytes())
    rc=json.loads((path.parent/'receipt.json').read_bytes())
    if rc['seed']!=a['seed'] or rc['arm']!=a['arm']:raise ValueError(path)
    e={p['event']:p['windows_monotonic_ns']/1.e9 for p in a['events']}
    if list(e.values())!=sorted(e.values()):raise ValueError('Reversed event chronology')
    c=a['cross_clock_bracket']
    offset=((c['windows_before_ns']+c['windows_after_ns'])/2-c['wsl_monotonic_ns'])/1.e9
    e.update(first_action=a['first_action_wsl_monotonic_ns']/1.e9+offset,
             last_action=a['final_action_wsl_monotonic_ns']/1.e9+offset)
    return dict(id=path.parent.name,lane=a['lane'],arm=a['arm'],scope=a['scope'],containers=a['containers'],events=e,
                clock_error_bound_s=c['maximum_midpoint_error_ns']/1.e9,
                phase_timing_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                bytes_transferred=sum((path.parent/f).stat().st_size for f in ('decisions.tsv','native.tsv','observations.u8')))

def report(roots):
    records=[read(p) for q in roots for p in sorted(q.glob('*/phase-timing.json')) if (p.parent/'receipt.json').is_file()]
    if not records:raise ValueError('No phase-instrumented original receipts')
    signed=defaultdict(list)
    for q in roots:
        for p in q.glob('*/receipt.json'):
            rc=json.loads(p.read_bytes())
            if not rc.get('lane') or not (p.parent/'decisions.tsv').is_file():continue
            with (p.parent/'decisions.tsv').open() as fp:
                columns=fp.readline().strip().split('\t')
                first=fp.readline().strip().split('\t')
            signed[rc['lane']].append((int(first[columns.index('onset_ns')]),p.parent.name))
    consecutive={(a[1],b[1]) for seq in signed.values() for a,b in zip(sorted(seq),sorted(seq)[1:])}
    grouped=defaultdict(list)
    phases={}
    for r in records:
        grouped[r['lane']].append(r)
        phases[r['id']]={name:r['events'][end]-r['events'][start] for name,start,end in COMPONENTS
                          if start in r['events'] and end in r['events']}
    pairs=[]
    for lane,rr in grouped.items():
        rr.sort(key=lambda x:x['events']['first_action'])
        for a,b in zip(rr,rr[1:]):
            A,B=a['events'],b['events']
            residual=B['scheduler_entered_credit']-A['compressed_hashed_and_sealed']
            pairs.append(dict(previous=a['id'],following=b['id'],lane=lane,previous_and_following_arm=[a['arm'],b['arm']],
                        previous_terminal_and_seal_s=A['compressed_hashed_and_sealed']-A['last_action'],
                        release_to_scheduler_entry_s=residual,following_entry_to_first_native_input_s=B['first_action']-B['scheduler_entered_credit'],
                        full_gap_s=B['first_action']-A['last_action'],adjacent_signed_credits_verified=(a['id'],b['id']) in consecutive,
                        game_container_reused=a['containers'][0]==b['containers'][0],
                        same_actor_service_and_start_time=a['containers'][1]==b['containers'][1]))
    per={k:stats([x[k] for x in phases.values() if k in x]) for k,_,_ in COMPONENTS if any(k in x for x in phases.values())}
    complete=[p for p in pairs if p['adjacent_signed_credits_verified']]
    boundary={field:stats([p[field] for p in complete]) for field in ('full_gap_s','previous_terminal_and_seal_s','release_to_scheduler_entry_s','following_entry_to_first_native_input_s')} if complete else {}
    return dict(method='Each interval uses same-process monotonic time or Linux native input timestamps mapped through a per-credit round-trip midpoint. Timing uncertainty applies only at cross-OS endpoints. Adjacent signed receipts are checked against complete lane inventories in the supplied study roots. Scheduler residual may include model readiness or waiting, and is not attributed to either. Native gameplay process and policy service have distinct lifetimes.',
                count=len(records),lanes=sorted(grouped),summaries=per,clock_roundtrip_halfwidth_s=stats([r['clock_error_bound_s'] for r in records]),
                transferred_bytes=dict(mean=statistics.mean(r['bytes_transferred'] for r in records),minimum=min(r['bytes_transferred'] for r in records),maximum=max(r['bytes_transferred'] for r in records)),
                verified_adjacent_pairs=len(complete),boundary_summaries=boundary,pairs=pairs,per_credit=phases,
                sources=[{k:v for k,v in r.items() if k!='events'} for r in records])

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',action='append',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=report(args.root)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'count':result['count'],'lanes':result['lanes'],'timings':result['summaries']},indent=2))
