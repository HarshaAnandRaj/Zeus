import sys
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import json
import pathlib
import re
from collections import Counter

import corpus.build_corpus_v2 as b
from tokenizers import Tokenizer

OUT = pathlib.Path('zeus_sandbox/universe/reports/pilot_audit.json')

# ---- rebuild the pilot (same code path emit() uses) ---------------------
manifest, kept = b.build(include_wiki=False)
print('ACCEPT total tokens:', sum(d['tokens'] for d in kept), 'docs:', len(kept))

R = {}

# 1. per-source accepted tokens
src = b._by_source(kept)
R['per_source_tokens'] = {k: v for k, v in src.items()}

# 2. FineWeb-Edu score distribution (score >= 3 tranche focus)
# re-read the raw jsonl to recover scores (kept docs store filtered text)
fw_docs = [d for d in kept if str(d['source']).startswith('fw_edu_batch')]
print('fineweb kept docs:', len(fw_docs))
from collections import defaultdict
score_by_batch = defaultdict(Counter)
score_all = Counter()
score_tokens = Counter()
tok = Tokenizer.from_file(str(b.TOKENIZER))
for d in fw_docs:
    # score is not stored on kept rec; recover from source line is lost.
    # re-derive by re-parsing the raw jsonl line content is complex; instead
    # re-read jsonl by matching raw_sha1.
    srcname = d['source']
    srcname = srcname.split(':')[0] if ':' in srcname else srcname
print('NOTE: score not carried on kept records; recomputing from jsonl below')

# Recompute score distribution + per-tranche metrics by re-reading jsonl and
# running the same filters per record.
score_dist = Counter()
score_markup = Counter()
score_nurl = Counter()
score_tok = Counter()
tranche_samples = defaultdict(list)
for jpath in sorted((b.RAW / 'fineweb_edu').glob('*.jsonl')):
    for line in jpath.read_text(encoding='utf-8', errors='replace').splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except (ValueError, TypeError):
            continue
        sc = int(obj.get('score') or 0)
        text = (obj.get('text') or '')
        score_dist[sc] += 1
        score_tok[sc] += len(text) // 4
        nurl = len(b.RE_URL.findall(text))
        if nurl and len(text) / nurl < b.MAX_URL_RATIO:
            score_nurl[sc] += 1
        markup = sum(1 for ch in text if ch in b.MARKUP_CHARS)
        if markup / max(len(text), 1) > b.MAX_MARKUP_FRAC:
            score_markup[sc] += 1
        if len(tranche_samples[sc]) < 3 and sc >= 3:
            tranche_samples[sc].append(text[:400])

R['fineweb_score_dist_docs'] = {str(k): v for k, v in sorted(score_dist.items())}
R['fineweb_score_dist_estrawtok'] = {str(k): v for k, v in sorted(score_tok.items())}
R['fineweb_score_markup_rej_docs'] = {str(k): v for k, v in sorted(score_markup.items())}
R['fineweb_score_url_rej_docs'] = {str(k): v for k, v in sorted(score_nurl.items())}
R['fineweb_score_samples'] = {str(k): v for k, v in sorted(tranche_samples.items())}

# 3. dup / near-dup: rejection histogram + cross-source exact-dup overlap
R['reject_histogram'] = manifest['accounting']['reject_histogram']

# 5. doc-level train/val separation: tokenize val, check token overlap not-in-train
# assign splits identically to emit()
rng = __import__('random').Random(20260903)
b._assign_split(kept, 0.02, rng)
train_docs = [d for d in kept if d['split'] == 'train']
val_docs = [d for d in kept if d['split'] == 'val']
print('train docs', len(train_docs), 'val docs', len(val_docs))
train_sets = set()
for j in range(0, len(train_docs), 200):
    chunk = [d['filtered'] for d in train_docs[j:j + 200]]
    for enc in tok.encode_batch(chunk):
        train_sets.update(enc.ids)
val_uniq = set()
n_val_four = 0
in_train4 = 0
sample_four = []
for d in val_docs:
    ids = tok.encode(d['filtered']).ids
    if len(ids) < 4:
        continue
    for i in range(len(ids) - 3):
        four = tuple(ids[i:i + 4])
        val_uniq.add(four)
        n_val_four += 1
        if set(ids[i:i+4]) <= train_sets:
            in_train4 += 0
# compare val 4-grams to an in-memory train fingerprint (sampled)
import numpy as np
print(f'val unique 4-grams: {len(val_uniq)} ; sampled train unigram set size {len(train_sets)}')
R['val_docs'] = len(val_docs)
R['train_docs'] = len(train_docs)

# tokenizer unknown / fragmentation on a representative val slice
unk_id = None
for tid, tokc in enumerate(tok.get_vocab().items()):
    if tokc[0] == '<unk>':
        unk_id = tid
        break
# get_vocab returns {token: id}; find <unk>
vocab = tok.get_vocab()
inv = {v: k for k, v in vocab.items()}
unk_id = vocab.get('<unk>')
n_unk = 0
n_tok = 0
for d in val_docs[:300]:
    ids = tok.encode(d['filtered']).ids
    n_tok += len(ids)
    if unk_id is not None:
        n_unk += ids.count(unk_id)
R['tokenizer_unk_rate'] = (n_unk / n_tok) if n_tok else 0
R['tokenizer_unk_id'] = unk_id
R['vocab_size'] = len(vocab)

# fragmentation proxy: avg bytes/token on val slice (lower bytes/token ~ more fragmented)
nbytes = 0
for d in val_docs[:300]:
    nbytes += len(d['filtered'].encode('utf-8'))
R['tokenizer_avg_bytes_per_token'] = round(nbytes / max(n_tok, 1), 3)
R['tokenizer_avg_chars_per_token'] = round(sum(len(d['filtered']) for d in val_docs[:300]) / max(n_tok,1), 3)

# 7. representative samples per source stratum (already have fineweb; add gutenberg + dialogue)
samples = {}
for sname in ['gutenberg_emerson_essays.txt', 'gutenberg_darwin_beagle_voyage.txt',
              'gutenberg_sherlock_holmes_adventures.txt', 'dailydialog.txt',
              'gutenberg_wells_time_machine.txt']:
    slist = [d for d in kept if d['source'] == sname]
    if slist:
        samples[sname] = [d['filtered'][:300] for d in slist[:2]]
R['source_samples'] = samples

(OUT).write_text(json.dumps(R, indent=2, ensure_ascii=False), encoding='utf-8')
print('audit written ->', OUT)
print('fineweb score dist (docs):', dict(sorted(score_dist.items())))
