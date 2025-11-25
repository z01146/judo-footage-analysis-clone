matrix_match_nomatch = {
    # TP for class(i): We predict as class i and its true.
    # TN for class(i): We predict as any class NOT i and its true.
    # FP for class(i): We predict as class i and it is NOT true. i.e. ground truth is some other class than (i)
    # FN for class(i): We predict as any class NOT i and it is NOT true for class (i), i.e. ground truth is class(i)
    "match": {
        "tp": 353,
        "tn": 327,
        "fp": 5,
        "fn": 7,
    },
    "match_intro": {
        "tp": 80,
        "tn": 599,
        "fp": 0,
        "fn": 2,
    },
    "match_outro": {
        "tp": 42,
        "tn": 637,
        "fp": 0,
        "fn": 0,
    },
    "no_match": {
        "tp": 204,
        "tn": 475,
        "fp": 8,
        "fn": 4,
    },
}

for k in matrix_match_nomatch:
    # TP for class(i): We predict as class i and its true.
    # TN for class(i): We predict as any class NOT i and its true.
    # FP for class(i): We predict as class i and it is NOT true. i.e. ground truth is some other class than (i)
    # FN for class(i): We predict as any class NOT i and it is NOT true for class (i), i.e. ground truth is class(i)

    tp = matrix_match_nomatch[k]["tp"]
    tn = matrix_match_nomatch[k]["tn"]
    fp = matrix_match_nomatch[k]["fp"]
    fn = matrix_match_nomatch[k]["fn"]
    a = (tp + tn) / (tp + tn + fp + fn)
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = (2 * p * r) / (p + r)
    print(
        f"{k}:\nF1-score={f1:.4f} Accuracy={a:.4f} Precision={p:.4f} Recall={r:.4f}\n"
    )
