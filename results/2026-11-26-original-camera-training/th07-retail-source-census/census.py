"""Read embedded retail patch identity and stage slots from byte prefixes of registered replays."""
from pathlib import Path
import hashlib,struct,re,unicodedata,json,threading,collections
from concurrent.futures import ThreadPoolExecutor
import requests
w=Path('work/th07-retail-census')
o=Path('outputs/touhou-vision-261126/th07-retail-source-census')
d=o.parent
tls=threading.local()
def rk(p):return json.loads(p.read_bytes())
def save(p,ob):p.write_text(json.dumps(ob,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def norm(x):return ' '.join(unicodedata.normalize('NFKC',x).casefold().strip().split())
def prefix_decompress(data,n=148):
 at=0;out=bytearray();window=bytearray(8192)
 def bits(count):
  nonlocal at
  if at+count>8*len(data):raise ValueError('short compressed prefix')
  v=0
  for unused in range(count):
   v=(v<<1)|((data[at//8]>>(7-at%8))&1);at+=1
  return v
 while len(out)<n:
  if bits(1): run=[bits(8)]
  else:
   adr=bits(13)
   if adr==0:raise ValueError('early terminator')
   count=bits(4)+3
   run=(window[(adr-1+k)&8191]for k in range(count))
  for val in run:window[len(out)&8191]=val;out.append(val)
 return bytes(out[:n])
def decode(b):
 assert len(b)>=148
 x=bytearray(b);assert x[:6]==bytes.fromhex('543752500011')
 x[16:]=bytes((val-x[13]-7*i)%256 for i,val in enumerate(x[16:]))
 declared,comp,exp=struct.unpack_from('<3I',x,16);assert 0<comp<4_000_000 and 148<=exp<8_000_000
 st=struct.unpack_from('<7I',x,28);slow=struct.unpack_from('<7I',x,56)
 sub=prefix_decompress(x[84:])
 marker=sub[140:146].split(b'\0')[0].decode('ascii',errors='replace')
 major=struct.unpack_from('<H',sub,22)[0];minor=chr(sub[1]);ver='unknown'
 if major==256 and minor in ('_','a','b') and marker=='0100'+minor:ver='1.00'+('' if minor=='_'else minor)
 occupied=[]
 for i,start in enumerate(st,1):
  if not start:continue
  assert 84+148<=start<84+exp
  end=min(v for v in [*st,*slow,84+exp] if v>start)
  assert end-start>44
  occupied.append(i)
 return dict(version=ver,embedded_marker=marker,major=major,minor_byte=sub[1],pilot=sub[10:18].decode('cp932',errors='replace').strip(' \x00'),
    rank=sub[3],shot=sub[2],stages=occupied,recorded_exe_size=int.from_bytes(sub[132:136],'little'),recorded_exe_checksum=sub[136:140][::-1].hex(),source_reported_score=int.from_bytes(sub[24:28],'little')*10)
def inspect(row):
 try:
  if not hasattr(tls,'s'):tls.s=requests.Session()
  with tls.s.get(row['url'],headers={'Range':'bytes=0-1023'},timeout=10,stream=True)as resp:
   resp.raise_for_status();b=next(resp.iter_content(1024))
  if b[:6]!=bytes.fromhex('543752500011'):return {**row,'other_header':b[:6].hex()}
  return {**row,**decode(b), 'prefix_sha256':hashlib.sha256(b).hexdigest(),'prefix_size':len(b)}
 except Exception as e:return {**row,'error':str(e) or type(e).__name__}
def main():
 files=rk(d/'th07-normal-source-audit/catalogue-native-build-audit.json')['records']
 full=(Path('work/th07-human-expansion/gensokyo-all.src')).read_text('utf8')
 table={}
 import html
 from urllib.parse import urljoin
 for tr in re.findall(r'<tr[^>]*>(.*?)</tr>',full,re.S):
  cells=re.findall(r'<td[^>]*>(.*?)</td>',tr,re.S)
  if len(cells)<8:continue
  link=re.search(r'href="([^"]+\.rpy)"',cells[-1])
  if link:table[urljoin('https://maribelhearn.com',link[1])]=re.findall('alt="([^"]+)"',cells[5])
 items=[dict(url=r['url'],archive='Gensokyo',author=r['player'],category=r['type'],conditions=table.get(r['url'],[]))for r in files if r['first8'].startswith('543752500011')]
 for r in rk(d/'th07-human-author-expansion/additional-catalogues.json')['rows']:
  if r['site']in ('Silent Selene','Royalflare')and r.get('header')=='543752500011':
   items.append(dict(url=r['url'],archive=r['site'],author=r['archive_author'],category=r['section'],conditions=[]))
 assert len(items)==556
 with ThreadPoolExecutor(6)as pool:result=list(pool.map(inspect,items))
 for j,record in enumerate(result):
  if 'error'in record:result[j]=inspect(items[j])
 save(o/'retail-prefixes.json',result)
 print('complete',len(result),'errors',sum('error' in r for r in result),collections.Counter(r.get('version')for r in result))
if __name__=='__main__':main()
