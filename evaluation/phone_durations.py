"""Reproduce Table 4 (durations of dog phones and noise labels).

Input is the frame-level HuBERT k-means transcription (one sentence per line,
one label per 20 ms frame), e.g. data/dog-phone/train.km produced in step 4.
A run of identical consecutive labels is one phone instance.

Usage: python phone_durations.py path/to/train.km
"""
import sys

import numpy as np

FRAME_MS = 20  # HuBERT label rate is 50 Hz

# k-means clusters judged to be noise (see Figure 3)
NOISE_LABELS = {4, 10, 20, 23, 24, 32, 37, 39, 50, 52, 53, 54, 65, 71, 72, 78, 79,
                82, 85, 87, 89, 108, 109, 114, 119, 120, 122, 134, 137}


def phone_runs(labels):
    prev, length = labels[0], 1
    for label in labels[1:]:
        if label == prev:
            length += 1
        else:
            yield prev, length
            prev, length = label, 1
    yield prev, length


def main(km_file):
    durations = {"Dog": [], "Noise": [], "All": []}
    with open(km_file) as f:
        for line in f:
            labels = line.split()
            if not labels:
                continue
            for label, length in phone_runs(labels):
                ms = length * FRAME_MS
                durations["Noise" if int(label) in NOISE_LABELS else "Dog"].append(ms)
                durations["All"].append(ms)

    print(f"{'':<6}{'Mean':>10}{'Median':>10}{'Std':>10}")
    for name, values in durations.items():
        v = np.array(values)
        print(f"{name:<6}{v.mean():>8.1f}ms{np.median(v):>8.0f}ms{v.std():>8.1f}ms")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/dog-phone/train.km")
