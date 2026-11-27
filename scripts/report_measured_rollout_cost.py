"""Compare measured original-game throughput with verified archive sizes and learner work."""
import argparse
import json
from pathlib import Path

from scripts.checkpoint_storage import digest


def read(path):return json.loads(path.read_bytes())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--public',type=Path,required=True)
    public=parser.parse_args().public;study=public/'six-game-return-horizon'
    out=study/'engineering-costs';out.mkdir(exist_ok=True)
    learn=read(study/'six-gae985/policy/report.json');elapsed=read(public/'four-title-complete-epochs/epoch-20/report.json')['history']
    train=read(study/'th17-train-actual-rollout-throughput.json');held=read(study/'th17-held-actual-rollout-throughput.json')
    arms={}
    for title in sorted({v['title'] for v in learn['credits']}):
        credits=[v for v in learn['credits'] if v['title']==title];rows=sum(v['steps'] for v in credits)
        asset=read(study/f'th{title:02}-exact-optimizer-input.release.json')
        arms[f'TH{title:02}']=dict(credits=len(credits),decisions=rows,compressed_bytes=asset['bytes'],bytes_per_decision=asset['bytes']/rows,archive_sha256=asset['sha256'])
    total=sum(v['compressed_bytes'] for v in arms.values());n=sum(v['decisions'] for v in arms.values())
    if n!=learn['steps'] or len(elapsed)!=20:raise ValueError('Incomplete measurement')
    controls=sum(v['windows'] for v in elapsed);seconds=sum(v['wall_seconds'] for v in elapsed)
    page=['# Measured throughput, transfer and storage', '',
          '| Original game | Original credits | PPO decisions | Exact compressed learner input (MiB) | Archive bytes per decision |', '|---|---:|---:|---:|---:|']
    for name,v in arms.items():page.append(f"| {name} | {v['credits']} | {v['decisions']:,} | {v['compressed_bytes']/2**20:.1f} | {v['bytes_per_decision']:,.0f} |")
    page+=['',f'The complete six-game input occupies {total/2**20:.1f} MiB for {n:,} unique decisions. The exact archives include the per-game compressed trajectory payload and a copy of the full manifest. Original recordings and checkpoint downloads are accounted for separately. These numbers bound archive transport for one transfer per batch; they do not measure protocol round trips or per-step network traffic.', '',
           '| Four-instance TH17 worker measurement | Common-window length | Mean decisions/s with four active | Mean decisions/s including gaps | Median interval between credits |', '|---|---:|---:|---:|---:|']
    for label,v in [('Training',train),('Held evaluation',held)]:
        four=v['groups']['4'];end=v['end_to_end']
        page.append(f"| {label} | {end['physical_wall_seconds']:.1f} s | {four['mean_aggregate_four_head_decisions_per_second']:.1f} | {end['amortized_decisions_per_second']:.1f} | {end['intercredit_gap_seconds_median']:.1f} s |")
    page+=['',f'On Momentum, the RTX 5090 traversed the 257,145-window human camera corpus twenty times. The {controls:,} training presentations took {seconds:.1f} seconds summed over the twenty training loops ({controls/seconds:,.0f} presentations/s). That timer excludes staging, replay decoding, held inference, checkpoint writing and original gameplay. A new physical transition and a repeated optimization sample carry different information.', '',
           'The TH17 gap runs from the last applied action in one credit to the next first action on that instance. It includes menu reset, evidence collection, packing, model readiness and scheduling. The figures use the interval when four slots were allocated to this study. Separate native inputs retain stage, stock and actual key evidence.', '',
           '[Native training intervals](../th17-train-actual-rollout-throughput.json) · [Native held intervals](../th17-held-actual-rollout-throughput.json) · [Optimizer trace](../six-gae985/policy/report.json) · [Twenty-epoch learner trace](../../four-title-complete-epochs/epoch-20/report.json)']
    (out/'README.md').write_text('\n'.join(page)+'\n',encoding='utf8')
    (out/'costs.json').write_text(json.dumps(dict(by_title=arms,compressed_input_bytes=total,optimizer_decisions=n,sft_presentations=controls,sft_loop_seconds=seconds,
        ppo_report=digest(study/'six-gae985/policy/report.json'),sft_report=digest(public/'four-title-complete-epochs/epoch-20/report.json')),indent=2)+'\n')


if __name__=='__main__':main()
