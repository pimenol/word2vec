import numpy as np


SEMANTIC_CATS = {
    'capital-common-countries', 'capital-world', 'currency',
    'city-in-state', 'family',
}


def evaluate_analogies(W, word2idx, idx2word, categories, batch_sz=400):
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-8
    Wn = (W / norms).astype(np.float32)

    results = {}
    sem_c = sem_t = sem_s = syn_c = syn_t = syn_s = 0

    for cat, qs in categories.items():
        is_sem = cat in SEMANTIC_CATS
        answerable = [q for q in qs if all(w in word2idx for w in q)]
        skipped = len(qs) - len(answerable)
        correct = 0

        for i in range(0, len(answerable), batch_sz):
            batch = answerable[i:i + batch_sz]
            n = len(batch)
            ai = np.array([word2idx[q[0]] for q in batch])
            bi = np.array([word2idx[q[1]] for q in batch])
            ci = np.array([word2idx[q[2]] for q in batch])
            di = np.array([word2idx[q[3]] for q in batch])

            vec = Wn[bi] - Wn[ai] + Wn[ci]
            vnorm = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8
            vec /= vnorm
            sims = vec @ Wn.T   # (n, V)

            for j in range(n):
                sims[j, ai[j]] = -np.inf
                sims[j, bi[j]] = -np.inf
                sims[j, ci[j]] = -np.inf

            correct += int((np.argmax(sims, axis=1) == di).sum())

        total = len(answerable)
        acc = correct / total if total else 0.0
        results[cat] = dict(correct=correct, total=total, skipped=skipped,
                            accuracy=acc, is_semantic=is_sem)
        if is_sem:
            sem_c += correct; sem_t += total; sem_s += skipped
        else:
            syn_c += correct; syn_t += total; syn_s += skipped

    summary = {
        'semantic':  dict(correct=sem_c, total=sem_t, skipped=sem_s,
                          accuracy=sem_c / sem_t if sem_t else 0),
        'syntactic': dict(correct=syn_c, total=syn_t, skipped=syn_s,
                          accuracy=syn_c / syn_t if syn_t else 0),
        'overall':   dict(correct=sem_c+syn_c, total=sem_t+syn_t,
                          skipped=sem_s+syn_s,
                          accuracy=(sem_c+syn_c)/(sem_t+syn_t)
                          if sem_t+syn_t else 0),
    }
    return results, summary


def nearest_neighbors(W, idx2word, word2idx, words, k=10):
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-8
    Wn = W / norms
    res = {}
    for w in words:
        if w not in word2idx:
            res[w] = [("OOV", 0.0)]; continue
        i = word2idx[w]
        sims = Wn @ Wn[i]
        sims[i] = -np.inf
        top = np.argsort(sims)[-k:][::-1]
        res[w] = [(idx2word[j], float(sims[j])) for j in top]
    return res


def analogy_examples(W, idx2word, word2idx, queries, k=5):
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-8
    Wn = W / norms
    out = []
    for a, b, c in queries:
        if any(w not in word2idx for w in (a, b, c)):
            out.append((a, b, c, [("OOV", 0.0)])); continue
        vec = Wn[word2idx[b]] - Wn[word2idx[a]] + Wn[word2idx[c]]
        vec /= np.linalg.norm(vec) + 1e-8
        sims = Wn @ vec
        for w in (a, b, c):
            sims[word2idx[w]] = -np.inf
        top = np.argsort(sims)[-k:][::-1]
        out.append((a, b, c, [(idx2word[j], float(sims[j])) for j in top]))
    return out
