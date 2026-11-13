"""Word statistics for the top-100 / top-200 candidate words (Sec. 4.2).

Prints the mean and variance of word duration and the word-length
distribution (Figure 6). Words are ranked by occurrence count after removing
words in which at least half of the phones are noise labels.

Input is the word table written by
5_word_discovery/word_validation.py::generate_words_clips_adagram, or the
released data/words/word_occurrences.csv.gz (columns include word,
start_time, end_time).

Usage: python word_statistics.py data/words/word_occurrences.csv.gz
"""
import csv
import gzip
import sys
from collections import Counter

import numpy as np

from phone_durations import NOISE_LABELS


def is_noisy(word):
    phones = [int(p) for p in word.split("-")]
    return sum(p in NOISE_LABELS for p in phones) / len(phones) >= 0.5


def main(words_csv):
    opener = gzip.open if words_csv.endswith(".gz") else open
    with opener(words_csv, "rt", newline="") as f:
        rows = [(r["word"], float(r["start_time"]), float(r["end_time"])) for r in csv.DictReader(f)]
    counts = Counter(r[0] for r in rows)
    ranked = [w for w, _ in counts.most_common() if not is_noisy(w)]

    for top_n in (100, 200):
        top = set(ranked[:top_n])
        durations = [end - start for word, start, end in rows if word in top]
        lengths = Counter(len(w.split("-")) for w in top)
        print(f"Top {top_n}: mean duration {np.mean(durations):.3f}s, "
              f"variance {np.var(durations):.4f}, "
              f"word lengths {dict(sorted(lengths.items()))}")


if __name__ == "__main__":
    main(sys.argv[1])
