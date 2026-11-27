"""Read the reached Stage 1 digit on TH17's highlighted original result row."""
from pathlib import Path
import hashlib,json,argparse
import numpy as np
from PIL import Image

def yellow(path):
    im=np.asarray(Image.open(path).convert('RGB')).astype('i2');r,g,b=im.transpose(2,0,1)
    return (r>165)&(g>140)&(r-b>95)&(g-b>82)
def dice(a,b):
    return float(2*(a&b).sum()/max(1,int(a.sum()+b.sum())))
def inspect(path,ref):
    mask=yellow(path);scores=[]
    for y in range(90,272):
        label=dice(mask[y-6:y+7,335:384],ref['label'])
        if label<.85:continue
        digit=dice(mask[y-6:y+7,390:404],ref['one'])
        scores.append((label,y,digit))
    if not scores:return dict(stage=None,reason='No matching yellow score-ranking row')
    label,y,digit=max(scores)
    return dict(stage=1 if digit>=.88 else None,label_dice=label,digit_one_dice=digit,center_y=y)
def update(root,out,ref_path):
    rows=[]
    with np.load(ref_path) as ref:
        for receipt in sorted(root.glob('*-*-*/receipt.json')):
            credit=json.loads(receipt.read_bytes());p=receipt.parent/'terminal.png';score=inspect(p,ref)
            if score['stage'] is not None and score['stage']!=credit['maximum_stage']:raise ValueError(('Native and visual stage disagree',receipt,score))
            rows.append(dict(episode=receipt.parent.name,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),native_stage=credit['maximum_stage'],
                terminal_screen=credit['terminal_screen'],independent_original_ranking=score))
    out.write_text(json.dumps(dict(coverage=len(rows),original_stage_one_reads=sum(r['independent_original_ranking']['stage']==1 for r in rows),
        source='Yellow highlighted original score-ranking row and exact Stage 1 glyph, with separate shape match for Stage label and final digit. Without a matching row the optical read is unknown.',credits=rows),indent=2)+'\n')
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('output',type=Path);a=parser.parse_args()
    update(a.root,a.output,Path(__file__).with_name('score-stage-one.npz'))
