"""Reproduce Table 3 (phone listening test).

answer_sheet.txt has one line per audio pair: "<phone(s)>,<pair index>,<label>",
where label 1 = both clips are the same phone (140 pairs) and
label 0 = two different phones with close cluster centres (150 pairs).
tester1.txt / tester2.txt hold each tester's answer (1 = same, 0 = different),
one per line, in the same order.

Usage: python table3.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def read_answers(name):
    with open(os.path.join(HERE, name)) as f:
        return [int(line.strip().split(",")[-1]) for line in f if line.strip()]


def pct(hits, idx):
    return 100 * sum(hits[i] for i in idx) / len(idx)


def main():
    truth = read_answers("answer_sheet.txt")
    t1 = read_answers("tester1.txt")
    t2 = read_answers("tester2.txt")
    assert len(truth) == len(t1) == len(t2) == 290

    ap = range(len(truth))
    spp = [i for i in ap if truth[i] == 1]
    dpp = [i for i in ap if truth[i] == 0]

    print(f"{'':<10}{'AP':>8}{'SPP':>8}{'DPP':>8}")
    for name, pred in (("Tester 1", t1), ("Tester 2", t2)):
        hits = [p == t for p, t in zip(pred, truth)]
        print(f"{name:<10}{pct(hits, ap):>7.2f}%{pct(hits, spp):>7.2f}%{pct(hits, dpp):>7.2f}%")
    agree = [a == b for a, b in zip(t1, t2)]
    print(f"{'Agreement':<10}{pct(agree, ap):>7.2f}%{pct(agree, spp):>7.2f}%{pct(agree, dpp):>7.2f}%")


if __name__ == "__main__":
    main()
