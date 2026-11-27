"""Summarize measured TH12 rollout costs from the exact qualified optimizer population."""
import argparse
import json
from pathlib import Path
from statistics import median

from scripts.checkpoint_storage import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    root=p.parse_args().study.resolve()
    files=root/'native12/credits'
    selection=json.loads((root/'th12-train-source-selection.json').read_bytes())
    rows=[]
    for slot in selection['rows']:
        matches=list(files.glob('train-*-'+str(slot['selected_seed'])+'-'+slot['lane']+'.json'))
        if len(matches)!=1 or digest(matches[0])!=slot['receipt_sha256']:
            raise ValueError('Missing the signed physical training credit')
        r=json.loads(matches[0].read_bytes())
        t=r['engineering_timing']
        rows.append(dict(host=slot['host'],lane=slot['lane'],seed=slot['selected_seed'],receipt_sha256=slot['receipt_sha256'],
                         setup=t['setup_until_first_action_seconds'],game=t['native_game_seconds'],
                         remote_pack=t['remote_pack_seconds'],request_to_header=t['ssh_request_to_header_seconds'],
                         receive=t['archive_receive_seconds'],local_verification=t['local_unpack_verify_npz_seconds'],
                         raw_bytes=t['raw_directory_bytes'],wire_bytes=t['wire_gzip_bytes'],actions=t['native_decisions']))
    def summarize(samples):
        keys=('setup','game','remote_pack','request_to_header','receive','local_verification')
        return dict(physical_credits=len(samples),median_seconds={k:median(r[k] for r in samples) for k in keys},
                    bytes_on_wire=sum(r['wire_bytes'] for r in samples),uncompressed_bytes=sum(r['raw_bytes'] for r in samples),
                    measured_receive_MiB_per_s=sum(r['wire_bytes'] for r in samples)/2**20/sum(r['receive'] for r in samples))
    result=dict(scope='All 16 TH12 credits admitted to this optimizer. Timings come from the production acquisition path. Remote compression occurs within the request-to-header interval. The receive throughput excludes request startup. Competing lane workloads and the three excluded technical credits are outside these per-credit summaries.',
                overall=summarize(rows),by_host={host:summarize([r for r in rows if r['host']==host]) for host in sorted({r['host'] for r in rows})},rows=rows)
    (root/'th12-qualified-lane-costs.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['overall'],flush=True)


if __name__=='__main__':
    main()
