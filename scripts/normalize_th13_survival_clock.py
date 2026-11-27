"""Replace TH13 wall-hold survival reward with within-stage, native-clock credit."""
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
from scripts.checkpoint_storage import digest


def native_progress(raw):
    # Only count previously unseen script ticks traversed by this action in one stage.
    peaks={}
    awaiting_reset={}
    increments=np.zeros(len(raw),dtype=np.int32)
    for i,(before,after,left,right) in enumerate(raw[:,[125,126,134,135]].astype(np.int64)):
        if before == after and before > 0 and left >= 0 and right >= 0:
            if after in awaiting_reset:
                # The original can publish Stage 2 one tick before its Stage 1
                # timer resets. Ignore the outgoing clock until the reset.
                outgoing=awaiting_reset[after]
                if left>=outgoing and right>=left:
                    awaiting_reset[after]=right
                    continue
                del awaiting_reset[after]
                if right<left:
                    peaks[after]=right
                    continue
                peaks[after]=left
            start=max(left,peaks.get(after,left))
            increments[i]=max(0,right-start)
            peaks[after]=max(start,right)
        elif after > 0 and right >= 0:
            if before>0 and after>before and left>=0 and right>=left:
                awaiting_reset[after]=right
            else:
                peaks[after]=max(peaks.get(after,0),right)
    return increments


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();base=a.original/'camera-batch';a.out.mkdir(parents=True,exist_ok=True)
    manifest=base/'credits.json';rows=json.loads(manifest.read_bytes());audit=[]
    for row in rows:
        source=base/row['file'];target=a.out/row['file']
        if digest(source)!=row['sha256']:raise ValueError('Original batch changed')
        if row['title']!=13:
            if not target.exists():
                try:os.link(source,target)
                except OSError:shutil.copyfile(source,target)
            if digest(target)!=row['sha256']:raise ValueError('Control title changed')
            continue
        (receipt,)=list((a.original/'native13').glob(f"train-*-{row['seed']}/receipt.json"))
        record=json.loads(receipt.read_bytes())
        if digest(receipt)!=row['source_log_sha256'] or record['role']!='train':raise ValueError('Wrong native source')
        raw=np.fromfile(receipt.parent/'steps.bin',dtype='<f4').reshape(-1,143)
        with np.load(source,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        if not np.array_equal(raw[:,119],arrays['reward']) or not np.array_equal(raw[:,110:114],arrays['action']):
            raise ValueError('Camera batch and original step trace disagree')
        original=-raw[:,136]+raw[:,137]+2*raw[:,138]+.001*raw[:,140]-.03*raw[:,139]
        np.testing.assert_allclose(original,arrays['reward'],atol=2e-7,rtol=0)
        ticks=native_progress(raw)
        wall=raw[:,140]*.001
        survival=ticks.astype(np.float32)*(.001/60.)
        arrays['reward_original']=arrays['reward']
        arrays['native_ticks_earned']=ticks
        arrays['reward']=(arrays['reward_original']-wall+survival).astype(np.float32)
        np.savez_compressed(target,**arrays)
        audit.append(dict(seed=row['seed'],host=record['host'],decisions=len(ticks),native_ticks=int(ticks.sum()),
                          old_survival=float(wall.sum()),new_survival=float(survival.sum()),raw_steps_sha256=digest(receipt.parent/'steps.bin'),
                          packed_before=row['sha256'],packed_after=digest(target)))
        row['sha256']=audit[-1]['packed_after']
    summary={}
    for host in sorted({r['host'] for r in audit}):
        group=[r for r in audit if r['host']==host];ticks=sum(r['native_ticks'] for r in group)
        summary[host]=dict(credits=len(group),ticks=ticks,
            original_reward_per_1000_ticks=sum(r['old_survival'] for r in group)*1000/ticks,
            corrected_reward_per_1000_ticks=sum(r['new_survival'] for r in group)*1000/ticks)
    (a.out/'credits.json').write_text(json.dumps(rows,indent=2)+'\n')
    (a.out/'clock-audit.json').write_text(json.dumps(dict(reference_manifest=digest(manifest),host=summary,
        credits=audit,formula='0.001 per 60 previously unseen native script ticks traversed within a stage; other event rewards retained.'),indent=2)+'\n')


if __name__=='__main__':
    main()
