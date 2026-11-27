"""Measure movement-change recall and false alerts for the three fixed replay models."""
import argparse
import json
from pathlib import Path

import torch

from replays.action_prior import ActionPrior, VOCAB_FIELDS
from replays.fit_five_title_action_rehearsal import roster, evaluate
from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data','tokens','cache','historical','parent'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    out = args.data/'five-title-human-action-rehearsal'
    stream, _, held, _ = roster(args)
    report=json.loads((out/'report.json').read_bytes())
    protocol=json.loads((out/'protocol.json').read_bytes())
    vocab=json.loads((args.parent/'vocab.json').read_bytes())
    for key in ('title','stage'):
        vocab[key]={int(k):v for k,v in vocab[key].items()}
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    model=ActionPrior([len(vocab[k])+1 for k in VOCAB_FIELDS]).to(device)
    groups={}
    for arm,path,sha in (('mixed',out/'epoch-4.pt',report['final_sha256']),
                         ('th08_only',args.data/'th08-human-actions/action-prior/epoch-4.pt',protocol['th08_only_sha256']),
                         ('frozen_mean',args.parent/'prior.pt',protocol['parent_sha256'])):
        if digest(path)!=sha:
            raise ValueError('Changed comparison model')
        model.load_state_dict(torch.load(path,map_location=device,weights_only=True))
        if arm=='frozen_mean':
            with torch.no_grad():
                for field in ('title','shot'):
                    w=model.metadata[VOCAB_FIELDS.index(field)].weight
                    w[0].copy_(w[1:].mean(0))
        data=evaluate(model,stream,held,vocab,device,turn_detection=True)
        for cell,people in data.items():
            for person,row in people.items():
                historical=report['by_author'][arm][cell][person]
                if row['next_direction']!=historical['next_direction'] or row['correct_changes']!=historical['correct_changes']:
                    raise ValueError('Held predictions differ from the initial evaluation')
        groups[arm]=data
    freeze(out/'turn-detection.json',dict(report_sha256=digest(out/'report.json'),
           definitions='An alert predicts any change from the previous human direction. A detected turn coincides with an actual change even if the new direction differs. An exact turn also matches the new direction.',by_arm=groups))


if __name__=='__main__':
    main()
