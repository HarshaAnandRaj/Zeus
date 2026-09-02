"""Quick corpus stats."""
import numpy as np
from collections import Counter

ids = np.load("C:/Users/Anand/Desktop/Projects/Zeus/corpus/data/train_ids.npy").tolist()
print(f"Tokens: {len(ids)}")

freq = Counter(ids)
print(f"Vocab used: {len(freq)}/8192")
sf = sorted(freq.values(), reverse=True)
print(f"Top token: {sf[0]}  10th: {sf[9]}  ratio: {sf[0]/sf[9]:.1f}")

repeats = sum(1 for i in range(1, len(ids)) if ids[i] == ids[i-1])
print(f"Adjacent repeats: {repeats/(len(ids)-1)*100:.2f}%")

total = len(ids)
unigram_e = -sum((f/total)*np.log(f/total) for f in freq.values())
print(f"Unigram entropy: {unigram_e:.4f} nats  perplexity: {np.exp(unigram_e):.1f}")

# Bigram stats on sample
s = ids[:200000]
bigram = {}
for i in range(len(s)-1):
    k = s[i]
    v = s[i+1]
    if k not in bigram: bigram[k] = Counter()
    bigram[k][v] += 1

surps = []
for i in range(1, len(s)):
    k, t = s[i-1], s[i]
    if k in bigram:
        tot = sum(bigram[k].values())
        f = bigram[k].get(t, 0)
        surps.append(-np.log(f/tot) if f > 0 else 15)
    else:
        surps.append(15)
surps = np.array(surps)
print(f"Bigram entropy: {surps.mean():.4f} nats")
print(f"Surp>3: {(surps>3).mean()*100:.1f}%  >5: {(surps>5).mean()*100:.1f}%  >8: {(surps>8).mean()*100:.1f}%")
