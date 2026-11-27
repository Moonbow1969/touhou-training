"""Join a completed camera SFT checkpoint's actual training inventory to verified replay metadata."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from scripts.checkpoint_storage import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data','base','expanded','report','out'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    report=json.loads(a.report.read_bytes())
    base=json.loads(a.base.read_bytes())['split']['train']
    added=json.loads(a.expanded.read_bytes())['records']
    inputs={r['sha256']:r['file'] for r in base}
    inputs.update({r['archive_sha256']:r['path'] for r in added})
    if len(inputs)!=len(report['train']) or set(inputs)!={r['sha256'] for r in report['train']}:
        raise ValueError('Source set differs from completed optimizer run')
    cells=defaultdict(list)
    details=[]
    for r in report['train']:
        file=a.data/inputs[r['sha256']]
        meta=json.loads(file.with_suffix('.json').read_bytes())
        if digest(file)!=r['sha256'] or meta['title']!=r['title']:raise ValueError('Training shard changed')
        difficulty=(meta['difficulty'] if 'difficulty' in meta else meta['mode'].split()[-1]).title()
        if difficulty not in ('Easy','Normal','Hard','Lunatic','Extra'):raise ValueError(difficulty)
        row=dict(title=r['title'],difficulty=difficulty,pilot=r['group'],stage=meta['stage'],shot=meta['shot'],
                 usable_decisions=r['decisions'],camera_sha256=r['sha256'],source_sha256=meta['replay_sha256'])
        details.append(row);cells[(r['title'],difficulty)].append(row)
    summary=[dict(title=t,difficulty=l,windows=sum(r['usable_decisions'] for r in rows),
                  stage_shards=len(rows),pilots=len({r['pilot'] for r in rows}),shots=sorted({r['shot'] for r in rows}))
             for (t,l),rows in sorted(cells.items())]
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(dict(checkpoint_sha256=report['torchscript_sha256'],report_sha256=digest(a.report),
        count_unit='Unique eligible human camera/action windows in completed SFT train shards. Repeated optimizer presentations excluded. Pilot counts may overlap between cells.',summary=summary,sources=details),indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
