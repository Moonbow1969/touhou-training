"""Check native TH07 build bytes in public Normal Reimu A replay headers."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import html,json,re,urllib.request
from hashlib import sha256
from collections import Counter
from urllib.parse import urljoin
import requests
import threading

root=Path(__file__).resolve().parent
out=root.parents[1]/'outputs/touhou-vision-261126/th07-normal-source-audit'
out.mkdir(exist_ok=True)
raw=(root/'gensokyo-normal-a.html').read_bytes()
native=(root.parents[1]/'outputs/touhou-vision-261126/th07-original-normal/official-engineering.rpy').read_bytes()
records=[]
for tr in re.findall(r'<tr[^>]*>(.*?)</tr>',raw.decode(),re.S):
    cells=re.findall('<td[^>]*>(.*?)</td>',tr,re.S);links=re.findall(r'href="([^"]+\.rpy)"',tr,re.I)
    if len(cells)<7 or not links or cells[1]!='PCB Normal<br>ReimuA':continue
    clean=lambda text:html.unescape(re.sub(r'<[^>]+>',' ',text)).strip()
    records.append(dict(url=urljoin('https://maribelhearn.com/',links[0]),player=clean(cells[0]),comment=clean(cells[6]),type=clean(cells[4])))
tls=threading.local()
def probe(r):
    if not hasattr(tls,'session'):tls.session=requests.Session()
    for i in range(2):
        try:
            with tls.session.get(r['url'],headers={'Range':'bytes=0-31'},stream=True,timeout=12) as response:
                response.raise_for_status();part=next(response.iter_content(32))
            r['first8']=part[:8].hex()
            r['same_version_bytes']=part[:6]==native[:6]
            if r['same_version_bytes']:
                b=tls.session.get(r['url'],timeout=20);b.raise_for_status()
                p=root/('candidate-'+r['url'].split('/')[-2]+'.rpy')
                p.write_bytes(b.content);r['sha256']=sha256(b.content).hexdigest()
            return r
        except Exception as e:r['error']=type(e).__name__
    return r
with ThreadPoolExecutor(5) as pool:rows=list(pool.map(probe,records))
doc=dict(source='https://maribelhearn.com/gensokyo?game=PCB&diff=Normal&shot=ReimuA',
         index_sha256=sha256(raw).hexdigest(),native_reference_first8=native[:8].hex(),scope='Returned archive entries with exact displayed category and shot',
         n=len(rows),matched=[r for r in rows if r.get('same_version_bytes')],headers=dict(Counter(r.get('first8','error')[:12]for r in rows)),records=rows)
(out/'catalogue-native-build-audit.json').write_text(json.dumps(doc,indent=2)+'\n')
print(doc['n'],doc['headers'],len(doc['matched']))
