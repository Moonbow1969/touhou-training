"""Separate replay action quantization from the deployed model's one-step decision error."""
from importlib import import_module
import json
from pathlib import Path

import numpy as np
import torch

bench=import_module('th10-retail-latency')
from scripts.train_human_camera_policy import load_replay
from scripts.pretrain_policy import movement_direction

device='cuda' if torch.cuda.is_available() else 'cpu'
torch.set_num_threads(2)
choices={
    'five-title-sft':bench.data.parent/'first-five-title-retail-sft/five-reviewed/policy.ts',
    'six-title-ppo':bench.data.parent/'chained-six-game-ppo/policy/policy.ts',
    'normal-specialist-sft':bench.data/'normal-sft/policy.ts',
}
rows=[]
for model_name,path in choices.items():
    actor=torch.jit.load(str(path),map_location=device).eval()
    for stage in (1,2):
        file=bench.data.parent/f'human-cross-title-v2/held/th10-gensokyo-2749-s{stage}.npz'
        source=load_replay(file)
        with np.load(file,allow_pickle=False) as z:
            original=movement_direction(z['input_chunk'].astype('i8').reshape(-1)).reshape(-1,4)
        if len(original)!=len(source['raw']):
            with np.load(file,allow_pickle=False) as z:
                clock=z['native_frame']
                accept=(np.diff(clock,prepend=clock[0])==4)
                if 'stable' in z:accept &= z['stable'].astype(bool)
            original=original[accept]
        y=source['action'][:,0]
        if len(original)!=len(y) or not np.array_equal(y,original[:,0]):
            raise ValueError('Held original row selection does not match model labels')
        with torch.inference_mode():
            inputs=torch.from_numpy(source['raw']).to(device).float()/255
            score=torch.cat([actor(x)[0].softmax(-1) for x in inputs.split(128)]).cpu().numpy()
        greedy=score.argmax(-1)
        sampled=float(np.sum(1-score[np.arange(len(y))[:,None],original]))
        rows.append(dict(arm=model_name,stage=stage,image_windows=len(y),game_ticks=original.size,
            actor_sha256=bench.base.digest(path),source_sha256=bench.base.digest(file),
            ideal_start_command_hold_tick_errors=int(np.sum(original!=y[:,None])),
            actual_argmax_hold_tick_errors=int(np.sum(original!=greedy[:,None])),
            expected_sampled_hold_tick_errors=sampled,
            first_tick_argmax_accuracy=float(np.mean(y==greedy))))
result=dict(player='Duckator',setting='Steam retail TH10 Normal Reimu A',
            method='Fixed originally visited human screens. Every arm predicts one 4-tick block using the same images. Each mismatch compares to the genuine input at the same native tick. The ideal first-command hold is a control-bandwidth reference. Model argmax or expected sampled command errors add action-selection error. This measures imitation on a single held player, not autonomous performance.',rows=rows)
(bench.study/'held-policy-vs-bandwidth.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
for r in rows:
    print(r['arm'],r['stage'],r['image_windows'],*[round(r[k]/r['game_ticks']*100,2) for k in
        ('ideal_start_command_hold_tick_errors','actual_argmax_hold_tick_errors','expected_sampled_hold_tick_errors')])
