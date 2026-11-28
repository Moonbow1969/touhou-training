import hashlib,json,sys,zipfile
from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np
root=Path('outputs/touhou-vision-261126/th07-normal-source-audit')
r=json.loads((root/'policy/report.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
native=r['native']
n0,n1=native
for meta in native:
    p=root/meta['role']/'paired.npz'
    assert sha(p)==meta['compressed_sha256']
    with np.load(p) as x:
        assert x['action'].shape==(meta['four_tick_pairs'],4)
        assert x['observation'].shape==(len(x['action']),2,112,96)
out=Image.new('RGB',(720,320),'white');draw=ImageDraw.Draw(out)
for i,meta in enumerate(native):
    path=root/meta['role']/'matched-midstage.png'
    pic=Image.open(path).convert('RGB').crop((0,0,640,480)).resize((352,264))
    out.paste(pic,(4+i*360,30))
    draw.text((8+i*360,8),('TRAIN | Boredperson','HELD | zylo')[i],fill='black')
    draw.text((8+i*360,295),f"{meta['four_tick_pairs']:,} exact 4-tick pairs",fill='black')
out.save(root/'original-stage1-witnesses.jpg',quality=90)
def f(d,k):
    if k=='four_head_joint_nll':return f"{d[k]:.4f}"
    return f"{100*d[k]:.1f}%"
a,b=r['held_before'],r['held_after']
rows=['| Held human action metric | Shared source | TH07 SFT |','|---|---:|---:|']
for h,name in enumerate(['Direction','Focus','Shoot','Bomb']):
    rows.append(f"| {name} cross-entropy (nats) | {a['nll_by_head'][h]:.4f} | {b['nll_by_head'][h]:.4f} |")
    rows.append(f"| {name} top-1 | {a['accuracy_by_head'][h]:.2%} | {b['accuracy_by_head'][h]:.2%} |")
for k,name in [('four_head_joint_nll','Sum of four cross-entropies'),('movement_change_acc','Direction on expert changes'),('onset_acc','Direction on stationary-to-moving onsets')]:
    rows.append(f"| {name} | {f(a,k)} | {f(b,k)} |")
report=f"""# TH07 Normal replay imitation

The official Touhou 7 v0.11 trial replayed two Normal Reimu A recordings on one Momentum host. The public category listed 327 recordings. Two matched the trial input format and both were accepted by the original Replay menu. The remaining 325 candidates carried the retail build bytes. [Replay catalogue](https://maribelhearn.com/gensokyo?game=PCB&diff=Normal&shot=ReimuA) · [Official trial](https://www16.big.or.jp/~zun/html/th07_update.html)

| Source | Stage | 4-tick training/evaluation pairs | Current keys equal replay word | Previous keys equal preceding word |
|---|---:|---:|---:|---:|
| Boredperson, TRAIN | 1 | {n0['four_tick_pairs']:,} | {n0['same_tick_key_matches']:,}/{n0['n_native_checked']:,} | {n0['previous_tick_key_fraction']:.2%} |
| zylo, HELD | 1 | {n1['four_tick_pairs']:,} | {n1['same_tick_key_matches']:,}/{n1['n_native_checked']:,} | {n1['previous_tick_key_fraction']:.2%} |

At native stage tick t the current 16-bit controller word agreed with replay[t]. The archive records a screen read between equal native clock reads, the downsampled X11 image and the absolute change since t-4. Direction, Focus and Shoot predict t+1. Bomb predicts a rising edge in t+1 through t+4. The renderer can present on its own refresh cycle. The clock bracket validates the sampled game state, without individually measuring display-buffer latency.

The learner fine-tuned the shared six-title actor for two uniform passes of the training source using Adam at 0.00001. All four action heads shared the objective. Full optimizer history and weight hashes are in policy/report.json.

{chr(10).join(rows)}

There were {b['n_change']} held expert changes and {b['n_onset']} held movement onsets. Predicting the preceding direction on every sample gives {b['previous_direction_baseline_acc']:.2%}. The held clip contains {b['expert_bomb_positive']} decision windows with a Bomb press. Per-author action metrics support only this two-source Stage 1 pilot. The policy has not been tested in live play in this experiment.

Source credits and original files are linked in pilot-protocol.json. The two replay entries carry different uploader names. Identity beyond that catalogue attribution was not verified. The camera archive includes derived gameplay observations and telemetry only. The original game executable, data and third-party replay files are available from their respective sources.
"""
(root/'README.md').write_text(report,encoding='utf8')
z=root/'th07-normal-two-replay-observations.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as file:
    for name in ('pilot-protocol.json','policy/report.json','README.md','original-stage1-witnesses.jpg'):
        file.write(root/name,name)
    for role in ('train','held'):
        for p in sorted((root/role).iterdir()):
            file.write(p,p.relative_to(root),compress_type=zipfile.ZIP_STORED if p.suffix in ('.png','.npz') else zipfile.ZIP_DEFLATED)
with zipfile.ZipFile(z)as f:assert f.testzip() is None
print(sha(z),z.stat().st_size)
