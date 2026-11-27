"""Run a counterbalanced TH10 original 2×2 model-training-lag and control-duration comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path

from scripts.collect_th10_full_action import pin,install_weights,start_camera_service,one,save
from scripts.run_th10_camera_cohort import checkpoint,verify,digest


def run(args):
    root=args.root.resolve();base=root/'cadence-native10';base.mkdir(exist_ok=True)
    host='accelerator';client=json.loads(args.workers.read_bytes())[host]
    seeds=list(range(759001,759009))
    specs={'four-120':('four',120),'four-62':('four',62),'multi-120':('multi',120),'multi-62':('multi',62)}
    arms=list(specs)
    williams=[[0,1,3,2],[1,2,0,3],[2,3,1,0],[3,0,2,1]]
    assignment={s:[arms[i] for i in williams[j%4]] for j,s in enumerate(seeds)}
    paths={'four':root/'camera-sft-expanded/policy.ts','multi':root/'camera-sft-multilag/policy.ts'}
    plan=dict(host=host,seeds=seeds,order=assignment,concurrency=args.concurrency,
              design='Each of eight fresh controller seeds plays each of two six-human-player SFT models at two exact native holds, 62 ms and 120 ms. Williams order is replicated twice, balancing every arm by period and every ordered pair of immediate predecessors. Original Normal TH10 trial, fresh stock, all four actions, no assistance, no Continue. No mid-hold images. Native RNG and incidental load uncontrolled.',
              training_models={name:str(path.relative_to(root)) for name,path in paths.items()},
              settings={a:dict(model=m,hold_ms=h) for a,(m,h) in specs.items()})
    save(base/'protocol.json',plan)
    pin_record=pin(client,camera=True)
    networks={};checks={}
    import torch
    torch.set_num_threads(2)
    for model,path in paths.items():
        networks[model]=checkpoint(path);checks[model]=digest(path)
        install_weights(client,path,checks[model]);start_camera_service(client,pin_record,checks[model])
    assert len(set(checks.values()))==2
    for a in arms:(base/a).mkdir(exist_ok=True)
    receipts=[]
    for period in range(4):
        with ThreadPoolExecutor(args.concurrency) as pool:
            tasks={}
            # Interleave the two Williams replicates, then rotate lane order by period.
            order=[0,5,2,7,4,1,6,3];order=order[period:]+order[:period]
            for index in order:
                s=seeds[index];a=assignment[s][period];m,h=specs[a]
                task=(client,pin_record,checks[m],host,s,a,base/a,h,False)
                tasks[pool.submit(one,task)]=(s,a)
            failures=[]
            for f in as_completed(tasks):
                s,a=tasks[f]
                try:r=f.result()
                except Exception as error:
                    failures.append(dict(seed=s,arm=a,error=repr(error)));continue
                assert r['requested_hold_ms']==specs[a][1] and r['policy_sha256']==checks[specs[a][0]]
                receipts.append(dict(seed=s,arm=a,period=period+1,steps=r['steps'],native_outcome=r['native_outcome']))
                save(base/'progress.json',dict(complete=len(receipts),planned=32,receipts=receipts))
            if failures:
                save(base/'technical-failures.json',failures);raise RuntimeError(failures)
    for a,(m,h) in specs.items():
        verify(base/a,paths[m],networks[m],seeds,'held',host=host)
    from scripts.summarize_th10_cadence_factorial import summarize_factorial
    summarize_factorial(base)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=Path,required=True)
    p.add_argument('--concurrency',type=int,default=3)
    run(p.parse_args())
