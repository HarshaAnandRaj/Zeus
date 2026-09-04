import sys
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import corpus.build_corpus_v2 as b

m, kept = b.build(include_wiki=False)
src = b._by_source(kept)
tot = sum(s['tokens'] for s in src.values())
ndoc = sum(s['docs'] for s in src.values())
print('ACCEPT total tokens:', tot, ' docs:', ndoc)
print('reject hist:', m['accounting']['reject_histogram'])
print('\nper-source (tokens desc):')
for s, v in sorted(src.items(), key=lambda kv: -kv[1]['tokens'])[:46]:
    print(f"  {s:<58} docs={v['docs']:>5} tokens={v['tokens']:>8}")
