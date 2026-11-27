"""Check the TH08 action-only prior against persistence, neutral metadata and historical author overlap."""
import argparse
from html import unescape
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import torch

from replays.action_prior import ActionPrior, VOCAB_FIELDS
from replays.fit_th08_action_prior import pack, score
from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze


def identity(name):
    return unicodedata.normalize('NFKC', name).casefold().strip()


def old_credit_lookup(directory, index):
    lookup = {}
    for title, file_sha in index['source_credit_scoreboard_sha256'].items():
        f = directory / f'th{title}-index.html'
        if digest(f) != file_sha:
            raise ValueError('Changed historical author scoreboard')
        for row in re.findall(r'<tr[^>]*>(.*?)</tr>', f.read_text(encoding='utf8'), re.S):
            link = re.search(r'href="(/replays/royalflare/th[0-9]+/[^"]+\.rpy)"', row)
            fields = re.findall(r'<td[^>]*>(.*?)</td>', row, re.S)
            if link and len(fields) == 9:
                lookup[Path(link[1]).name] = identity(unescape(re.sub('<[^>]*>', '', fields[6])))
    return lookup


def old_credits(directory, index):
    lookup = old_credit_lookup(directory, index)
    old = json.loads((directory / 'expanded-corpus.json').read_bytes())
    source = [r for r in old if r['split'] == 'train']
    if len(source) != 1738 or any(r['name'] not in lookup for r in source):
        raise ValueError('Incomplete parent training-author assignments')
    return {lookup[r['name']] for r in source}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source','cache','parent','historical','author-audit'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    study = a.source / 'action-prior'
    rep = json.loads((study / 'report.json').read_bytes())
    raw = json.loads((a.source / 'source-manifest.json').read_bytes())
    stream, spans = pack(raw['qualified'], a.cache)
    author_index = json.loads(a.author_audit.read_bytes())
    old = old_credits(a.historical, author_index)
    vocab = json.loads((a.parent / 'vocab.json').read_bytes())
    for key in ('stage', 'title'):
        vocab[key] = {int(k): v for k, v in vocab[key].items()}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    m = ActionPrior([len(vocab[k])+1 for k in VOCAB_FIELDS]).to(device)
    m.load_state_dict(torch.load(a.parent / 'prior.pt', map_location=device, weights_only=True))
    with torch.no_grad():
        for key in ('title','shot'):
            embedding = m.metadata[VOCAB_FIELDS.index(key)].weight
            embedding[0].copy_(embedding[1:].mean(0))
    neutral = score(m, stream, spans, vocab, device)
    groups = {}
    for cohort, people in (('all', list(neutral)), ('absent_from_parent_train', [n for n in neutral if n not in old]),
                           ('also_in_parent_train', [n for n in neutral if n in old])):
        if not people:
            groups[cohort] = {'authors': []}
            continue
        fields = {}
        for field in ('next_direction','correct_changes','joint_nll'):
            values = {key: np.array([arm[n][field] for n in people]) for key, arm in
                      (('frozen_unknown_ids', rep['per_author']['frozen']), ('frozen_mean_metadata', neutral),
                       ('four_epoch_th08', rep['per_author']['epoch-4']))}
            if field == 'next_direction':
                values['repeat_last'] = np.array([neutral[n]['repeat_direction'] for n in people])
            if field == 'correct_changes':
                values['repeat_last'] = np.zeros(len(people))
            contrasts = {}
            for control in ('frozen_mean_metadata', 'repeat_last'):
                if control not in values:
                    continue
                dif = values['four_epoch_th08']-values[control]
                draws = np.random.default_rng(261129).choice(dif, (10000,len(dif)), replace=True)
                contrasts[control] = dict(mean=float(dif.mean()), author_bootstrap95=np.quantile(draws.mean(1),[.025,.975]).tolist())
            fields[field] = dict(means={name:float(x.mean()) for name,x in values.items()}, contrasts=contrasts)
        groups[cohort] = dict(authors=people, decisions=sum(neutral[n]['n'] for n in people), fields=fields)
    freeze(study / 'transfer-controls.json', dict(parent_author_ledger_sha256=digest(a.author_audit),
        th08_fit_report_sha256=digest(study / 'report.json'), known_parent_train_author_names=len(old),
        interpretation='Post-hoc diagnostics on the same sixteen distinct held TH08 credited names. '
                       'The historical title and shot ID zero received no historical training examples. '
                       'The frozen mean-metadata model substitutes the arithmetic means of the original known '
                       'title and shot embeddings and receives no TH08 gradient. Credited names are normalized by '
                       'Unicode NFKC and case folding. Alias identities not documented by the scoreboards remain unknown.',
        groups=groups, frozen_mean_metadata_by_author=neutral))


if __name__ == '__main__':
    main()
