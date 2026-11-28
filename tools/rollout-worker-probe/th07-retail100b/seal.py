"""Seal an original TH07 retail 1.00b human replay stage and build bounded causal windows."""
from pathlib import Path
import sys, struct, hashlib, json, gzip
import numpy as np
from PIL import Image
from inspect_rpy import decompress

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def source_info(path, stage):
    src=bytearray(path.read_bytes())
    assert src[:6]==bytes.fromhex('543752500011')
    src[16:]=bytes((v-src[13]-7*i)%256 for i,v in enumerate(src[16:]))
    clen,dlen=struct.unpack_from('<II',src,20)
    assert len(src)==84+clen
    data=decompress(bytes(src[84:]),dlen)
    off=struct.unpack_from('<14I',src,28);start=off[stage-1]
    end=min(v for v in [*off,84+len(data)] if v>start)
    score=int.from_bytes(data[24:28],'little')*10
    assert data[1:4]==b'b\x00\x01'and data[140:146]==b'0100b\x00'and int.from_bytes(data[22:24],'little')==256
    assert int.from_bytes(data[132:136],'little')==650752 and int.from_bytes(data[136:140],'little')==0xaec5445c
    real=np.frombuffer(bytes(data[start-84+44:end-84]),'<u2').reshape(-1,2)
    snapshot=np.frombuffer(bytes(data[start-84:start-84+44]),'u1')
    return real[:,0],snapshot,score


def direction(bits):
    word=bits.astype(np.int64)
    return 4 + ((word>>7)&1)-((word>>6)&1) + 3*(((word>>5)&1)-((word>>4)&1))


def seal(root, stage_dir, identity, source):
    protocol=json.loads((root/'protocol.json').read_bytes())
    rec=next(x for x in protocol['roles'] if x['id']==identity)
    p=stage_dir
    assert (p/'sensor.exit').read_text().strip()=='0'
    cap=json.loads((p/'capture.json').read_bytes());stage=cap['stage']
    assert sha(p/'th7_01.rpy')==rec['original_sha256']==cap['sha256']
    builds={l.split()[1]:l.split()[0] for l in (p/'build.sha256').read_text().splitlines()}
    assert builds=={'th07.exe':protocol['executable_sha256'],'th07.dat':protocol['data_sha256']}
    words,snapshot,score=source_info(p/'th7_01.rpy',stage)
    clock=np.loadtxt(p/'clock.csv',delimiter=',',dtype=np.int64,ndmin=2)
    t,now,previous,full,st,game_clock,flags,cursor,stage_base,manager=clock.T
    assert np.all(st==stage)and np.all(manager>0)and np.all(stage_base>0)
    assert t[0]<=12 and np.all(t%4==0) and np.all(np.diff(t)>0)
    assert abs(len(words)-t[-1])<=24,('terminal source clock',len(words),t[-1])
    assert np.all(cursor-stage_base==44+4*t)
    assert np.all(t<len(words)+1) and np.all(t>=2)
    assert np.array_equal(now,words[t-1]) and np.array_equal(previous,words[t-2])
    xy_struct=np.fromfile(p/'native.bin','<f4').reshape(-1,5,2)
    assert len(xy_struct)==len(t) and np.isfinite(xy_struct).all()
    xy=xy_struct[:,0].copy()
    e=np.stack([np.max(np.abs((xy_struct[:,a]+xy_struct[:,b])/2-xy),axis=1)for a,b in [(1,2),(3,4)]],1)
    good=np.all(e<.02,axis=1)&np.all((xy>=0)&(xy<=[384,448]),axis=1)
    with gzip.open(p/'gray.raw.gz','rb') as stream:gray=np.frombuffer(stream.read(),'u1').reshape(-1,112,96)
    assert len(gray)==len(t)
    ct=[]
    with gzip.open(p/'full-color.bgra.gz','rb')as stream:
        while raw_tick:=stream.read(4):
            tk=struct.unpack('<I',raw_tick)[0];ct.append(tk)
            original=Image.frombytes('RGB',(640,480),stream.read(640*480*4),'raw','BGRX')
            reduced=original.crop((32,16,416,464)).convert('L').resize((96,112),Image.Resampling.BOX)
            assert np.array_equal(gray[np.searchsorted(t,tk)],np.asarray(reduced))
    assert np.array_equal(ct,t[full==1])
    j=np.flatnonzero((np.diff(t)==4)&good[1:]&good[:-1]&(t[1:]+3<len(words)))+1
    first=words[t[j]].astype(np.int64)
    edge=np.diff(((words>>1)&1).astype(int),prepend=0)>0
    actions=np.column_stack([direction(first),(first>>2)&1,first&1,np.any(edge[t[j,None]+np.arange(4)],axis=1)]).astype('u1')
    pick=np.linspace(0,len(j)-1,min(2000,len(j))).astype(np.int64)
    assert len(np.unique(pick))==len(pick) and len(pick)>1000
    np.savez_compressed(p/'positions.npz',native_tick=t,xy_native=xy,xy_camera=xy/4,
                        native_word=now,previous_native_word=previous,stage_replay_words=words,
                        xy_qualified=good,causal_pair_rows=j,action_next_fourheads=actions,
                        fixed_optimizer_window_indices=pick,full_color_ticks=np.array(ct),
                        live_stage=st,original_current_word_index=t-1,retail_stage_start_snapshot=snapshot)
    files=['native.bin','clock.csv','candidates.tsv','positions.npz','full-color.bgra.gz','gray.raw.gz','source-list.png','stage-selector.png','selected-stage.png','playback-mode.png','after-stage.png','opening.png','build.sha256']
    doc=dict(author=rec['author'],source=rec['source'],role=rec['role'],stage=stage,rank='Normal',shot='reimua',original_replay_sha256=rec['original_sha256'],
             original_score_in_source=score,stage_source_records=len(words),rows=len(t),native_tick_range=[int(t[0]),int(t[-1])],
             native_current_equal=int((now==words[t-1]).sum()),native_previous_equal=int((previous==words[t-2]).sum()),
             cursor_equal_expected=int((cursor-stage_base==44+4*t).sum()),gray_regenerations=len(ct),
             native_gameclock_minus_replay_range=[int((game_clock-t).min()),int((game_clock-t).max())],excluded_xy_rows=int((~good).sum()),
             causal_four_tick_pairs=len(j),capped_windows=len(pick),nominal_four_tick_coverage=len(t)/(t[-1]/4),
             admitted=True,qualification='Original native stage ID, current/previous input and cursor for every measured decision. Black-box gameplay samples from the matching stock runtime. Includes sparse full-color camera witness checks.',
             protocol_sha256=sha(root/'protocol.json'),artifacts_sha256={n:sha(p/n)for n in files})
    (p/'source-audit.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
    print(rec['author'],stage,len(t),len(words),t[-1],len(j),len(pick),flush=True)
if __name__=='__main__':
    root=Path(__file__).resolve().parent.parent
    seal(root,Path(sys.argv[1]).resolve(),int(sys.argv[2]),None)
