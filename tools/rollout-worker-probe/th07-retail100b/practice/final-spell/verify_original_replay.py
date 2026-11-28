from pathlib import Path
import sys,time,subprocess,hashlib,csv,json,struct
import numpy as np
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def words(source):
 repo=next(parent for parent in Path(__file__).resolve().parents if (parent/'replays/inspect_rpy.py').is_file())
 sys.path.insert(0,str(repo/'replays'))
 from inspect_rpy import decompress
 blob=source.read_bytes();assert blob[:6]==bytes.fromhex('543752500011') and blob[-4:]==b'PRAC'
 s=bytearray(blob);s[16:84]=bytes((v-s[13]-7*i)%256 for i,v in enumerate(s[16:84]));c,d=struct.unpack_from('<II',s,20)
 s[84:84+c]=bytes((v-s[13]-7*(i+68))%256 for i,v in enumerate(blob[84:84+c]))
 data=decompress(bytes(s[84:84+c]),d);of=struct.unpack_from('<14I',s,28);a=of[0];end=min(v for v in [*of,84+len(data)] if v>a)
 assert data[140:146]==b'0100b\0'
 metadata=json.loads(blob[84+c:-8].split(b'\0')[0]);assert metadata['game']=='th07' and metadata['life']==2 and metadata['bomb']==3 and metadata['power']==128 and metadata['section']==6
 return np.frombuffer(bytes(data[a-84+44:end-84]),'<u2').reshape(-1,2)[:,0],metadata
def score(base,source):
 saved,meta=words(source)
 with (base/'native-input.csv').open() as f:
  rows=[r for r in csv.DictReader(f) if int(r['flags'])&4 and int(r['tick'])>=2 and int(r['tick'])==int(r['event_index']) and not int(r['retry'])]
 t=np.array([int(r['tick']) for r in rows]);cur=np.array([int(r['word']) for r in rows]);prev=np.array([int(r['previous_word']) for r in rows])
 offsets={}
 for off in range(-2,3):
  take=(t+off>=1)&(t+off<len(saved));ix=t[take]+off
  offsets[str(off)]=[int(len(ix)),int(np.sum((saved[ix]&247)==(cur[take]&247))),int(np.sum((saved[ix-1]&247)==(prev[take]&247)))]
 return dict(replay_file=source.name,replay_sha256=sha(source),native_sha256=sha(base/'native-input.csv'),phase_by_offset=offsets,replay_declared_thprac=meta,retail_input_words=len(saved))
if __name__=='__main__':
 for item in sys.argv[1:]:
  b=Path(item);src=next(b.glob('official-replays/*.rpy'));res=score(b,src);print(b.name,res['phase_by_offset']);(b/'official-replay-verified.json').write_text(json.dumps(res,indent=2)+'\n')

