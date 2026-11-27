"""Separate observed original endpoints from matched pairs with complete TH12 input coverage."""
import argparse
import json
from pathlib import Path

from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path)
    root=p.parse_args().root
    plan=json.loads((root/'camera-protocol.json').read_bytes())
    scores=root/'evaluation/credits.json'
    coverage=root/'th12-held-qualified-coverage.json'
    grades={v['receipt_sha256']:v for v in json.loads(coverage.read_bytes())['credits']}
    rows=[v for v in json.loads(scores.read_bytes()) if v['title']==12]
    if len(rows)!=len(grades) or {v['native_log_sha256'] for v in rows}!=set(grades):raise ValueError('Physical result and coverage mismatch')
    arms=(plan['reference_arm'],plan['candidate'])
    rank_cells={}
    for rank in (2,3):
        selected=[v for v in rows if v['difficulty']==rank]
        pairs={seed:{v['arm']:v for v in selected if v['seed']==seed} for seed in {v['seed'] for v in selected}}
        if any(set(cell)!=set(arms) for cell in pairs.values()):raise ValueError('Missing paired original')
        for row in selected:
            grade=grades[row['native_log_sha256']]
            if grade['coverage'] not in ('qualified','indeterminate') or not grade['evidence_sha256'] or grade['coverage']!=row.get('input_control_coverage','qualified'):
                raise ValueError('Conflicting native input qualification')
        complete=[seed for seed,cell in pairs.items() if all(grades[v['native_log_sha256']]['coverage']=='qualified' for v in cell.values())]
        def summarize(seeds):
            result={}
            for arm in arms:
                group=[pairs[s][arm] for s in seeds]
                result[arm]=dict(n=len(group),stage2=sum(v['stage2_reached'] is True for v in group),
                    unresolved_stage=sum(v['stage2_reached'] is None for v in group),
                    terminal_spares=[v['terminal_spares'] for v in group],terminal_bombs=[v['terminal_bombs'] for v in group])
            return dict(paired_seeds=sorted(seeds),arms=result)
        rank_cells[str(rank)]=dict(all_observed=summarize(sorted(pairs)),qualified_pairs=summarize(sorted(complete)))
    freeze(root/'evaluation/th12-physical-and-qualified.json',dict(original_score_sha256=digest(scores),coverage_manifest_sha256=digest(coverage),by_rank=rank_cells,
        interpretation='Every verified native opening and terminal counts in the physical population. The conditional subset includes a controller seed only when both arms have complete independently qualified control. Its selection may depend on achieved gameplay.'))


if __name__=='__main__':main()
