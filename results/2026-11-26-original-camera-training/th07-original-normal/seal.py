"""Seal original TH07 engineering qualification, OCR-independent inventory and movement evidence."""
from hashlib import sha256
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

out=Path(sys.argv[1])
def fileinfo(name):
    p=out/name
    return dict(file=name,bytes=p.stat().st_size,sha256=sha256(p.read_bytes()).hexdigest())
def pixels(name):
    return np.asarray(Image.open(out/(name+'.png')).convert('RGB'),dtype=np.int16)
def stocks(image):
    result=[]
    for y,life in [(103,True),(120,False)]:
        hits=[]
        for x in range(504,632,16):
            r,g,b=image[y-6:y+7,x-6:x+7].transpose(2,0,1)
            feature=(r>180)&(r-b>30) if life else (b>130)&(b-r>35)
            hits.append(int(feature.sum()))
        result.append(hits)
    return dict(spares=sum(n>=24 for n in result[0]),bombs=sum(n>=24 for n in result[1]),red=result[0],blue=result[1])
def player(image):
    roi=image[319:435,46:395]
    r,g,b=roi.transpose(2,0,1)
    y,x=np.where((r>160)&(r-g>75)&(r-b>75))
    return [int(np.median(x))+46,int(np.median(y))+319]
pulses=['focused-center','focus-left-1','focus-left-2','focus-right-1','focus-right-2','focused-up']
xy={p:player(pixels(p)) for p in pulses}
first=stocks(pixels('focused-center'));last=stocks(pixels('after-bomb'))
assert (first['spares'],first['bombs'],last['spares'],last['bombs'])==(2,3,2,2)
assert xy['focus-left-2'][0] < xy['focused-center'][0]-45 and xy['focus-right-2'][0] > xy['focus-left-2'][0]+45
assert xy['focused-up'][1]<xy['focus-right-2'][1]-20
document=dict(title='TH07 Perfect Cherry Blossom',edition='Official ZUN 0.11 trial, first three stages',host='momentum',
   source_page='https://www16.big.or.jp/~zun/html/th07_update.html',source='https://www16.big.or.jp/~zun/data/soft/youmu_tr011.lzh',
   package_sha256='b5414547b887ac0a7f2ee640713376af6440b96627bd2814fa680a0c62b5f344',
   game_sha256='2ca6b41ce2b07fb2b8a119f9683fc497dd724edcbc130b5076f8d86323b9091f',
   game_data_sha256='45bedfe287f5a577000146b3d8c0b57ceb213f342b2c08348afb50aa46a19099',
   docker_image='toho-trial:endeavor-native-v6',docker_image_id='sha256:299861299bf1090b56d2b3f198ef227644d89319520fb1f28cb5db344e77929d',
   recording='Private isolated Xvfb, Wine virtual desktop, software GL, null PulseAudio sink. Game is displayed at native 640x480. No RAM field, patch or trainer used.',
   role='Engineering-only physical input qualification',mode='Normal ordinary Start, Reimu A',opening=first,after_one_physical_x=last,
   position_method='Median red-sprite pixel in fixed empty opening lower field ROI; screen pixels, independent of requested keys.',
   original_xy=xy,terminal=dict(source='Original trial result screen manually read',score=18000,game_progress_pct=1.51,continues=0,misses=3,effective_bombs=1,spellcard_bonus_count=0,slowdown_pct=.46,stage2=False),
   evidence=[fileinfo(name+'.png') for name in ['title','menu1','menu2','menu3',*pulses,'live-shot','after-bomb','stock-paused','continue-default-no','settled-scorecard','score-replay-confirm-ReimuA']],
   sources=[fileinfo(name) for name in ['official-engineering.rpy','pulse.tsv','neutral-tail-real-time.mp4','launch.sh','pulse.sh','neutral-tail.sh']],
   video_scope='Time-sampled original RGB from resuming after the separate opening input test. The ordinary original ending, default No and subsequent menu are included.')
(out/'qualification.json').write_text(json.dumps(document,indent=2)+'\n')
