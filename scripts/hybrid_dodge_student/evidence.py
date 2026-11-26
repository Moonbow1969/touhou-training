"""Check whole-credit overlap, exact deployment parity and original 1CC archive coverage."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .actor import OUT,ROOT,StockHybridStudent,AnticipatoryContact,feature_vectors
from .prepare import SOURCE,WORLD,relative_actions
from .train import TRAIN
from scripts.checkpoint_storage import digest
from scripts.train_anticipatory_contact import AnticipatoryContact as OriginalEncoder
from scripts.train_local_avoidance import local_context


def fingerprints(path):
    with np.load(path,allow_pickle=False) as z:
        b=z['bullets'];c=z['context'][:,:10]
        if 'stage_frame' in z:
            f=z['stage_frame'];s=np.full(len(f),json.loads(path.with_suffix('.json').read_text())['stage'])
        else:
            names=json.loads(str(z['metadata_json']))['native_field_order'];a=z['native_before']
            f=a[:,names.index('stage_frame')];s=a[:,names.index('stage')]
    return [hashlib.blake2b(np.asarray([st,t],dtype='<i8').tobytes()+ctx.astype('<f4',copy=False).tobytes()
           +bul.astype('<f4',copy=False).tobytes(),digest_size=16).digest() for st,t,ctx,bul in zip(s,f,c,b)]


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    final=next(WORLD.glob('th14-relative-full-eval-frozen_student-3601*/transition-shards/*.npz'))
    original=set().union(*(fingerprints(SOURCE[s]) for s in TRAIN))
    duplicates={}
    for key,path in list((name,SOURCE[name]) for name in ['held4','held5','2402','2601','3301','3601'])+[('final',final)]:
        rows=fingerprints(path)
        duplicates[key]=dict(total_rows=len(rows),exact_train_rows=sum(r in original for r in rows),
                             unique_check_states=len(set(rows)),exact_train_shard_overlap=digest(path) in {digest(SOURCE[s]) for s in TRAIN})
    model=StockHybridStudent(OUT/'conservative.pt')
    legacy=OriginalEncoder().eval();legacy.load_state_dict(model.contact.state_dict())
    with np.load(final,allow_pickle=False) as z:
        b=z['bullets'];c=z['context'];before=z['native_before']
        cols={name:i for i,name in enumerate(json.loads(str(z['metadata_json']))['native_field_order'])}
        prior=relative_actions(model.model,before,cols,b,c)
        # Use the production predict entry point on every native row, including ineligible/terminal-adjacent rows.
        stage=before[:,cols['stage']];batches=[]
        encoder_difference=0.
        for i in range(0,len(b),256):
            bb,cc=torch.from_numpy(b[i:i+256]),torch.from_numpy(c[i:i+256])
            x=feature_vectors(model.contact,bb,cc,torch.tensor(stage[i:i+256]))
            cm=local_context(cc,0)
            encoder_difference=max(encoder_difference,float((model.contact(bb,cm)-legacy(bb,cm)).abs().max()))
            batches.append(model.movement.logits(x,torch.from_numpy(prior[i:i+256,0])).argmax(1).numpy())
        joint=np.concatenate(batches);expected=np.column_stack([joint%9,joint//9,prior[:,1:]])
        env=SimpleNamespace(reset_count=1)
        env.game=SimpleNamespace(read_game_val=lambda key:stocks if key=='bombs' else (_ for _ in ()).throw(KeyError(key)))
        model.environment=env;diff=np.zeros(4,dtype=int)
        for i,row in enumerate(before):
            read=lambda key:int(row[cols[key]])
            stocks=read('bombs');spell=read('spell_id')
            boss=SimpleNamespace(max_hp=read('boss_max_hp'),hp=read('boss_hp'),vulnerable=read('boss_vulnerable')==1) if read('boss_max_hp')>0 else None
            env.snapshot=SimpleNamespace(stage=read('stage'),spell_id=spell if spell>=0 else None,
                                         boss=boss,player_state=read('player_state'))
            env.held={'x'} if read('previous_x') else set();env.last_frame=read('stage_frame')
            actual=model.predict(dict(context=c[i],bullets=b[i]))[0]
            diff+=actual!=expected[i]
        base_four=np.column_stack([prior[:,0]%9,prior[:,0]//9,prior[:,1:]])
        old_match=(base_four==z['action']).all(1)
    if diff.any() or encoder_difference:
        raise RuntimeError(f'Inference parity issue: {diff}, {encoder_difference}')
    archive={}
    for src in (WORLD/'overnight-th14-analytic-story.json',WORLD/'overnight-th14-title-equal-story.json'):
        raw=json.loads(src.read_text(encoding='utf-8'))
        result=next(iter(raw['results'].values()))[0]
        archive[src.name]=dict(sha256=digest(src),outcome=result['outcome'],stages_cleared=result['stages_cleared'],
            decisions=result['steps'],misses=result['misses'],bombs_used=result['bombs_used'],sparse_boss_observations=len(result['observations']),
            impact_snapshots=len(result['impacts']),native_phases=len(result['phases']),raw_continues_field='native_continues' in result)
    for path in (WORLD/'experiment-ledger/operations-2026-11-23/overnight-final-recipe/1cc-aim-samples.npz',
                 WORLD/'overnight-th14-title-equal-story-aim-samples/1-analytic_dodge_with_learned_aim.npz'):
        with np.load(path) as z:
            archive[str(path.relative_to(WORLD))]=dict(sha256=digest(path),arrays={k:list(z[k].shape) for k in z.files})
    result=dict(exact_overlap=duplicates,hash_key='Native stage, native frame, float32 observed context[0:10] and all 96x7 projectile features. Base student/encoder have prior human-replay training. Files were separated by original credit. Similar but non-identical waves can recur.',
                full_size_parity_rows=len(b),production_vs_offline_mismatches=diff.tolist(),max_inherited_encoder_logit_difference=encoder_difference,
                full_native_incumbent_exact_rows=int(old_match.sum()),full_native_incumbent_rows=len(old_match),held_credit_sha256=digest(final),
                original_1cc_materials=archive)
    (OUT/'provenance-parity.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result)[:2170])


if __name__=='__main__':
    main()
