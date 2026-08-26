"""Two-schedule test (temporal ladder rung, pre-registered):
structural features of the token stream expect LOW CV (clockwork);
content features expect scatter. Verdicts: COUPLED-CLOCK / NESTED-INDEPENDENT /
SCHEDULED-EVERYWHERE (parrot in time domain) / FREE-FORM (no structural beat)."""
import torch

from common import load_model
from init_helper import seeded_reset

FUNC = {"the", "a", "an", "of", "to", "and", "it", "in", "that", "is",
        "was", "for", "on", "with", "as", "at", "by", "but", "or"}


def _cv(xs):
    if len(xs) < 3:
        return None
    arr = torch.tensor(xs, dtype=torch.float64)
    m = float(arr.mean())
    if m < 1e-9:
        return None
    return float(arr.std() / m)


def sentence_intervals(words):
    pos = [i for i, w in enumerate(words)
           if w and w.rstrip('",;:\'')[-1:] in ".!?"]
    return [b - a for a, b in zip(pos, pos[1:]) if b - a > 0]


def func_word_intervals(words):
    pos = [i for i, w in enumerate(words)
           if w.lower().strip('.,!?;:"\'') in FUNC]
    return [b - a for a, b in zip(pos, pos[1:]) if b - a > 0]


def content_novelty_cv(text, window=24):
    tris = [text[i:i + 3] for i in range(len(text) - 2)]
    seen = set()
    nov = []
    for t in tris:
        nov.append(0.0 if t in seen else 1.0)
        seen.add(t)
    rates = [sum(nov[i:i + window]) / window
             for i in range(0, max(len(nov) - window, 1), window)]
    return _cv(rates)


def generate(model, tokens=400, temp=0.8, seeds=(7, 13)):
    texts = []
    for seed in seeds:
        g = torch.Generator(device="cpu").manual_seed(seed * 31 + 7)
        seeded_reset(model, 0.05)
        logits = model.observe()
        out = []
        for _ in range(tokens):
            probs = torch.softmax(logits / max(temp, 1e-4), dim=-1)
            nxt = torch.multinomial(probs.cpu(), 1, generator=g).item()
            out.append(nxt)
            logits, _ = model.step(nxt)
        texts.append(model.decode(out))
    return texts


def main(model=None, tokens=400):
    model = model or load_model()
    all_sent, all_func, all_novel = [], [], []
    for text in generate(model, tokens=tokens):
        words = [w for w in text.replace("\n", " ").split(" ") if w]
        si = sentence_intervals(words)
        fi = func_word_intervals(words)
        nc = content_novelty_cv(text)
        all_sent.extend(si)
        all_func.extend(fi)
        if nc is not None:
            all_novel.append(nc)

    sent_cv = _cv(all_sent)
    func_cv = _cv(all_func)
    novel_cv = sum(all_novel) / len(all_novel) if all_novel else None
    struct_cv = min([c for c in (sent_cv, func_cv) if c is not None], default=None)

    if struct_cv is None:
        verdict = "FREE-FORM (no measurable structural beat; coupling test inconclusive)"
    elif struct_cv < 0.5 and novel_cv is not None and novel_cv > 0.6:
        verdict = "NESTED-INDEPENDENT (structural clockwork, content free — inside and outside)"
    elif struct_cv >= 0.7:
        verdict = "FREE-FORM (no structural schedule)"
    elif novel_cv is not None and novel_cv <= struct_cv * 1.5:
        verdict = "SCHEDULED-EVERYWHERE (parrot regime in time domain)"
    else:
        verdict = "COUPLED-CLOCK LEAN (structural beat present, content semi-free)"

    med_sent = sorted(all_sent)[len(all_sent) // 2] if all_sent else None
    return {"n_sentences": len(all_sent), "sent_interval_median": med_sent,
            "sent_cv": round(sent_cv, 4) if sent_cv else None,
            "func_cv": round(func_cv, 4) if func_cv else None,
            "content_novelty_cv": round(novel_cv, 4) if novel_cv is not None else None,
            "verdict": verdict}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
