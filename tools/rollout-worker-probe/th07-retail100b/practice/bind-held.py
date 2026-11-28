from pathlib import Path
import hashlib,json,sys
d=Path(sys.argv[1]);path=d/'paired-registration.json';reg=json.loads(path.read_bytes())
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if len(sys.argv)!=4:raise ValueError('Exact authorised SHA and candidate file required')
sha=sys.argv[2]
if h(Path(sys.argv[3]))!=sha:raise ValueError('Candidate SHA mismatch')
binding=dict(released=True,paired_registration_sha256=h(path),policies={'parent':reg['parent_policy_sha256'],'practice-ppo':sha},registered_arm_sha256={})
for arm,policy in binding['policies'].items():
 r=dict(role='held',policy_sha256=policy,original_schedule_sha256=h(path),arm=arm,rows=[dict(row,arm=arm) for row in reg['rows']])
 arm_path=d/f'arm-{arm}.json'
 with arm_path.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
 binding['registered_arm_sha256'][arm]=h(arm_path)
p=d/'released-binding.json'
with p.open('x') as f:json.dump(binding,f,indent=2);f.write('\n')
print(h(p),json.dumps(binding))
