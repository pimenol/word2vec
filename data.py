import collections
import os
import urllib.request
import zipfile

import numpy as np


def download_file(url, filepath):
    if os.path.exists(filepath):
        print(f"    {filepath} already exists, skipping.")
        return True
    print(f"    Downloading {url} ...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"    Saved to {filepath}")
        return True
    except Exception as e:
        print(f"    Download failed: {e}")
        return False


def load_text8(data_dir):
    """Load text8 corpus (first 100 MB of cleaned Wikipedia)."""
    txt_path = os.path.join(data_dir, 'text8')
    zip_path = os.path.join(data_dir, 'text8.zip')
    if not os.path.exists(txt_path):
        download_file('http://mattmahoney.net/dc/text8.zip', zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(data_dir)
    with open(txt_path, 'r') as f:
        text = f.read()
    return text.split()


def load_analogy_questions(data_dir):
    """Load Google analogy test set.  Returns dict category -> [(a,b,c,d)]."""
    filepath = os.path.join(data_dir, 'questions-words.txt')
    if not os.path.exists(filepath):
        download_file(
            'https://raw.githubusercontent.com/tmikolov/word2vec/master/questions-words.txt',
            filepath)
    categories = {}
    cur = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(':'):
                cur = line[2:]
                categories[cur] = []
            elif cur is not None:
                words = line.lower().split()
                if len(words) == 4:
                    categories[cur].append(tuple(words))
    return categories


def build_vocab(tokens, min_count):
    """Build vocabulary sorted by descending frequency, filtered by min_count."""
    counter = collections.Counter(tokens)
    vocab = [(w, c) for w, c in counter.most_common() if c >= min_count]
    word2idx = {w: i for i, (w, _) in enumerate(vocab)}
    idx2word = [w for w, _ in vocab]
    freqs = np.array([c for _, c in vocab], dtype=np.float64)
    return word2idx, idx2word, freqs


def tokens_to_ids(tokens, word2idx):
    """Convert token list to numpy array of integer IDs (skip OOV)."""
    ids = np.empty(len(tokens), dtype=np.int32)
    n = 0
    for w in tokens:
        idx = word2idx.get(w, -1)
        if idx >= 0:
            ids[n] = idx
            n += 1
    return ids[:n]