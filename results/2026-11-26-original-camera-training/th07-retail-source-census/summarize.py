import sys,copy,hashlib,subprocess,json,collections,shutil
from pathlib import Path
from census import w,o,save,rk,norm,decode
import requests
r=rk(o/'retail-prefixes.json')
profile=(650752,'aec5445c')
def ident(item):return norm(item['author'])
def verified_auth(item):
 return ident(item) not in ('','anonymous','guest','no name','unknown','n/a','?') and not any('tool' in t.lower()for t in item['conditions']+[item['category']])
usable=[item for item in r if item['rank']==1 and item['shot']==0 and {1,2}<=set(item['stages'])]
strict=[item for item in usable if verified_auth(item)]
pool=[item for item in strict if(item['recorded_exe_size'],item['recorded_exe_checksum'])==profile and item['version']=='1.00b' and item['pilot'].strip() and all(i in item['stages']for i in range(1,7))]
short=[];authors=set();pilots=set();prefixes=set()
for archive in ('Gensokyo','Silent Selene','Royalflare'):
 group=sorted([v for v in pool if v['archive']==archive],key=lambda v:hashlib.sha256(('TH07-retail-preflight-1/'+v['url']).encode()).hexdigest())
 found=0
 for v in group:
  if ident(v) in authors or norm(v['pilot']) in pilots or v['prefix_sha256'] in prefixes:continue
  short.append(v);authors.add(ident(v));pilots.add(norm(v['pilot']));prefixes.add(v['prefix_sha256']);found+=1
  if found==2:break
assert len(short)==6
baseline=sorted([v for v in strict if v['version'] in ('1.00','1.00a')],key=lambda v:v['version'])
keys_dir=w/'case-sources';keys_dir.mkdir(exist_ok=True)
paths=[]
for j,v in enumerate(short+baseline):
 f=keys_dir/f'public-{j+1:02d}.rpy'
 if not f.exists():
  response=requests.get(v['url'],timeout=16);response.raise_for_status();b=response.content
  assert 1024<len(b)<1_000_000
  f.write_bytes(b)
 b=f.read_bytes();assert hashlib.sha256(b[:v['prefix_size']]).hexdigest()==v['prefix_sha256']
 v['replay_sha256']=hashlib.sha256(b).hexdigest();paths.append(str(f))
src='work/sattori-reference/packages/replay-parser/src/games/th07.ts'
out=subprocess.check_output(['node',str(w/'independent.mjs'),src,*paths],encoding='utf8')
checks=json.loads(out)
for v,test in zip(short+baseline,checks):
 assert test['shot']=='ReimuA'and test['difficulty']=='Normal'and test['embedded_version']==v['embedded_marker']
 assert test['exe_bytes']==v['recorded_exe_size']and test['exe_checksum']==v['recorded_exe_checksum']
 assert test['archive_score']==v['source_reported_score'] and test['stages_in_independent_parser']==v['stages']
 assert all(x['complete_input_records']>200 and x['direction_records']>100 for x in test['stage_input_evidence'])
 v['ordinary_fire_candidate']=all(x['fire_records']>100 for x in test['stage_input_evidence'])
 v['full_independent_decode']=test
save(o/'shortlist.json',dict(selection='Two from each catalogue, ordered by SHA256(TH07-retail-preflight-1/ + original URL), unique named uploader, native pilot alias and prefix. All six originally report stage slots 1–6, correct shot/rank, local executable size/checksum. No runtime or training admission.',same_build_candidates=short,older_version_decoder_controls=baseline,external_decoder_sha256=hashlib.sha256(Path(src).read_bytes()).hexdigest(),upstream_git=subprocess.check_output(['git','-C','work/sattori-reference','rev-parse','HEAD'],text=True).strip()))
def counts(group):
 def get(g):
  return dict(catalogue_urls=len(g),distinct_normalized_uploader_handles=len({ident(v)for v in g}),distinct_native_pilot_strings=len({norm(v['pilot'])for v in g if norm(v['pilot'])}),distinct_1024byte_prefix_hashes=len({v['prefix_sha256']for v in g}))
 return dict(all=get(group),named_untagged_tools=get([v for v in group if verified_auth(v)]),matching_local_recorded_exe=get([v for v in group if verified_auth(v) and (v['recorded_exe_size'],v['recorded_exe_checksum'])==profile]))
stats={v:counts([item for item in usable if item['version']==v]) for v in ('1.00','1.00a','1.00b')}
save(o/'summary.json',dict(scanned=556,native_rank_shot_stage12=len(usable),exact_version=stats,across_versions=counts(usable),source_subset='Previously catalogued Normal Reimu A category at Gensokyo, Silent Selene and Royalflare. The two 0.11 trial matches were excluded up front.',independent_complete_replay_decodes=len(checks),strict_build_fingerprints=dict(collections.Counter(f"{v['recorded_exe_size']}/{v['recorded_exe_checksum']}" for v in usable if v['version']=='1.00b')),shortlist=[(v['author'],v['pilot'],v['url']) for v in short]))
print(json.dumps(stats,ensure_ascii=False))
print([(v['author'],v['pilot'],v['version'])for v in short+baseline])
