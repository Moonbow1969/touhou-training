"""Compare the last applied action against first final-life loss in paired native clock traces."""
import argparse
import csv
import json
from pathlib import Path

from scripts.checkpoint_storage import digest


def read(path):
    with path.open(encoding='utf8') as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    plan = json.loads((args.root / 'clock-protocol.json').read_bytes())
    for cell in [r for r in plan['plan'] if r['mode'] == 'train']:
        folder = args.root / cell['cohort'] / 'native12'
        receipt = folder / 'credits' / (cell['tag'] + '.json')
        credit = json.loads(receipt.read_bytes())
        raw = folder / 'raw' / cell['tag']
        native_file = raw / 'engineering-native.tsv'
        action_file = raw / 'engineering-decisions.tsv'
        quality = credit['control_clock_verification']
        if digest(native_file) != quality['native_trace_sha256'] or digest(action_file) != quality['decision_trace_sha256']:
            raise ValueError(f'Changed input evidence for {receipt}')
        actions = read(action_file)
        native = read(native_file)
        if len(actions) != credit['actions'] or any(int(r['action']) != i for i,r in enumerate(actions)):
            raise ValueError(f'Incomplete sequence of applied controls for {receipt}')
        death = next(r for r in native if r['mode']=='0' and r['stage']==r['manager_stage'] and int(r['spares']) < 0)
        if actions[-1]['post_stage'] != death['stage']:
            raise ValueError(f'Final action and final death occur in different stages for {receipt}')
        ticks = int(death['tick'])-int(actions[-1]['post_tick'])
        rows.append(dict(seed=cell['seed'], difficulty=cell['rank'], clock=cell['cohort'], host=cell['host'],
                         receipt_sha256=digest(receipt), actions=len(actions), last_control_post_tick=int(actions[-1]['post_tick']),
                         first_observed_final_death_tick=int(death['tick']), signed_tail_ticks=ticks,
                         last_native_tick=int(native[-1]['tick']), actual_consumed_keys=quality['native_key_matches'],
                         comparable_consumed_keys=quality['eligible_native_end_of_hold']))
    result=dict(definition='Original tick of first observed negative spare-life stock minus end tick of final policy decision. A positive interval measures time requiring investigation. It does not classify playability or exclude interior gaps. Study-specific photography adjudicates long tails.',
                protocol_sha256=digest(args.root/'clock-protocol.json'),credits=rows,
                over_60_native_ticks=[(r['clock'],r['seed'],r['signed_tail_ticks']) for r in rows if r['signed_tail_ticks']>60])
    (args.root/'terminal-input-coverage.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print('Audited',len(rows),'complete native credits; tails over 60 original ticks',result['over_60_native_ticks'],flush=True)


if __name__ == '__main__':
    main()
