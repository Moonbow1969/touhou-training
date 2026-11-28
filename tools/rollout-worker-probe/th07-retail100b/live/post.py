"""Navigate only a confirmed original terminal through the visible scorecard and replay save."""
from pathlib import Path
import json, shutil, subprocess, sys, time
import numpy as np
from PIL import Image
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'tools'))
from menu import snap,key

def score(im,ref,rect):
    def pixels(p):return np.asarray(Image.open(p).convert('L').crop(rect))>180
    a,b=pixels(im),pixels(ref)
    return float((a&b).sum()/max(1,(a|b).sum()))

def post(folder):
    q=json.loads((folder/'finished-game.json').read_bytes())
    name=(folder/'actual-container').read_text().strip() if (folder/'actual-container').exists() else f"moonbow-th07-retail-record-{q['seed']}"
    tool=root/'record/tool'
    witness=folder/'native-continue.png'
    def selected_no(path):
        rgb=np.asarray(Image.open(path).convert('RGB').crop((188,271,266,306)),dtype=np.int16)
        red,green,blue=rgb.transpose(2,0,1)
        return (red>110)&(red>green+40)&(red>blue+40)
    first,ref=selected_no(witness),selected_no(tool/'continue-ref.png')
    no=float((first&ref).sum()/max(1,(first|ref).sum()))
    assert no>.70,('Original negative Continue selection',no)
    # The two engineering lanes may have already reached ranking or the scorecard.
    for j in range(4):
        snap(name,f'pre-score-{j}')
        path=folder/f'pre-score-{j}.png'
        if score(path,tool/'score-ref.png',(55,125,184,299))>.80:
            shutil.copyfile(path,folder/'settled-scorecard.png');break
        key(name,'z',1.35)
    else:raise RuntimeError('Original after-credit statistics were not found')
    for j in range(12):
        key(name,'z',.72)
        snap(name,f'replay-save-{j:02d}')
        p=subprocess.run(['docker','exec',name,'sh','-c','test -s /tmp/th07/replay/th7_01.rpy && cp /tmp/th07/replay/th7_01.rpy /capture/official-policy.rpy'])
        if p.returncode==0:break
    else:raise RuntimeError('Original retail replay file not saved')
    settled=folder/'settled-scorecard.png'
    zero=score(settled,tool/'score-ref.png',(380,190,405,212))
    rank=score(settled,tool/'score-ref.png',(311,146,402,170))
    assert zero>.88 and rank>.88,(zero,rank)
    (folder/'ui-verified.json').write_text(json.dumps(dict(no=no,zero_continue=zero,normal_rank=rank,scorecard=True))+'\n')
    print('STOCK RPY',q['seed'],(folder/'official-policy.rpy').stat().st_size,flush=True)

if __name__=='__main__':post(Path(sys.argv[1]))
