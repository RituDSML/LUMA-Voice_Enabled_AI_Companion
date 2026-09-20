"""
evaluate_crisis_cosine.py — threshold sweep comparing THREE crisis
detection methods on the SAME 42-item labelled test set used in Chapter
7 (test_crisis_detection.py: 14 direct / 14 paraphrased / 14 negative
control), so all comparisons are on identical data:

  1. substring  — the existing safety.check_crisis() (unchanged baseline)
  2. cosine     — the new embedding-based check_crisis_cosine() alone
  3. hybrid     — substring OR cosine (check_crisis_hybrid()) — the
                  recommended production function, since it can only
                  ever catch as much or more than substring alone

Run: python evaluate_crisis_cosine.py

Outputs:
  - A per-threshold table (cosine-only and hybrid side by side, with the
    threshold-independent substring baseline printed once above it)
  - crisis_cosine_threshold_sweep.csv   (the full table, for Ch.7)
  - crisis_cosine_similarity_detail.csv (every utterance's max similarity
    score, closest tracked phrase, and each method's verdict — useful
    for spot-checking specific cases, the way CS-17/CS-09 were examined
    for RAG)
"""

import csv
from test_crisis_detection import DIRECT, PARAPHRASED, NEGATIVE_CONTROL
from safety import check_crisis
from crisis_detection_cosine import max_similarity

# A reasonably fine sweep — narrow this later around whatever range turns
# out to matter once you see the first pass.
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def compute_metrics(hits, direct_n, paraphrased_n, negative_n):
    """hits = {'direct': [bool,...], 'paraphrased': [...], 'negative': [...]}"""
    tp = sum(hits["direct"]) + sum(hits["paraphrased"])
    fn = (direct_n - sum(hits["direct"])) + (paraphrased_n - sum(hits["paraphrased"]))
    fp = sum(hits["negative"])
    tn = negative_n - fp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    direct_recall = sum(hits["direct"]) / direct_n
    paraphrased_recall = sum(hits["paraphrased"]) / paraphrased_n

    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "precision": precision,
            "recall": recall, "f1": f1, "direct_recall": direct_recall,
            "paraphrased_recall": paraphrased_recall}


def main():
    print("Loading embedding model and computing similarity for all 42 utterances")
    print("against the 40 tracked phrases in safety.py... (one-time model load, then fast)\n")

    scores = {"direct": [], "paraphrased": [], "negative": []}
    substr = {"direct": [], "paraphrased": [], "negative": []}
    detail = {"direct": [], "paraphrased": [], "negative": []}

    for cat, utterances in [("direct", DIRECT), ("paraphrased", PARAPHRASED), ("negative", NEGATIVE_CONTROL)]:
        for u in utterances:
            s, phrase = max_similarity(u)
            hit = check_crisis(u)
            scores[cat].append(s)
            substr[cat].append(hit)
            detail[cat].append((u, s, phrase, hit))

    direct_n, paraphrased_n, negative_n = len(DIRECT), len(PARAPHRASED), len(NEGATIVE_CONTROL)

    baseline = compute_metrics(substr, direct_n, paraphrased_n, negative_n)
    print("=" * 110)
    print("SUBSTRING baseline (safety.check_crisis(), unchanged — threshold-independent):")
    print(f"  Precision {baseline['precision']:.2f} | Recall {baseline['recall']:.2f} | F1 {baseline['f1']:.2f} | "
          f"Direct recall {baseline['direct_recall']:.2f} | Paraphrased recall {baseline['paraphrased_recall']:.2f} | "
          f"TP/FN/FP/TN = {baseline['tp']}/{baseline['fn']}/{baseline['fp']}/{baseline['tn']}")
    print("=" * 110)

    header = (f"{'Thresh':>7} | {'Cos P':>5} {'Cos R':>5} {'Cos F1':>6} {'CosDirR':>7} {'CosParR':>7} | "
              f"{'Hyb P':>5} {'Hyb R':>5} {'Hyb F1':>6} {'HybDirR':>7} {'HybParR':>7}")
    print("\n" + header)
    print("-" * len(header))

    results = []
    for t in THRESHOLDS:
        cosine_hits = {
            cat: [s >= t for s in scores[cat]] for cat in ("direct", "paraphrased", "negative")
        }
        hybrid_hits = {
            cat: [sub or (s >= t) for sub, s in zip(substr[cat], scores[cat])]
            for cat in ("direct", "paraphrased", "negative")
        }
        cos = compute_metrics(cosine_hits, direct_n, paraphrased_n, negative_n)
        hyb = compute_metrics(hybrid_hits, direct_n, paraphrased_n, negative_n)
        results.append({"threshold": t, "cosine": cos, "hybrid": hyb})

        print(f"{t:>7.2f} | {cos['precision']:>5.2f} {cos['recall']:>5.2f} {cos['f1']:>6.2f} "
              f"{cos['direct_recall']:>7.2f} {cos['paraphrased_recall']:>7.2f} | "
              f"{hyb['precision']:>5.2f} {hyb['recall']:>5.2f} {hyb['f1']:>6.2f} "
              f"{hyb['direct_recall']:>7.2f} {hyb['paraphrased_recall']:>7.2f}")

    perfect_precision = [r for r in results if r["hybrid"]["precision"] == 1.0]
    print()
    if perfect_precision:
        best = max(perfect_precision, key=lambda r: r["hybrid"]["paraphrased_recall"])
        h, c = best["hybrid"], best["cosine"]
        print(f"Recommended threshold (best HYBRID paraphrased recall at precision=1.00): {best['threshold']:.2f}")
        print(f"  Hybrid  -> Precision {h['precision']:.2f} | Recall {h['recall']:.2f} | F1 {h['f1']:.2f} | "
              f"Direct recall {h['direct_recall']:.2f} | Paraphrased recall {h['paraphrased_recall']:.2f}")
        print(f"  Cosine alone at this threshold -> Precision {c['precision']:.2f} | "
              f"Paraphrased recall {c['paraphrased_recall']:.2f}  (shown to isolate the semantic layer's own contribution)")
        print(f"  vs. substring baseline -> paraphrased recall {baseline['paraphrased_recall']:.2f} -> {h['paraphrased_recall']:.2f}, "
              f"precision held at {h['precision']:.2f}")
    else:
        best_f1 = max(results, key=lambda r: r["hybrid"]["f1"])
        h = best_f1["hybrid"]
        print("No threshold in the sweep held hybrid precision at 1.00.")
        print(f"Best hybrid F1 in sweep: threshold {best_f1['threshold']:.2f} "
              f"-> Precision {h['precision']:.2f}, Recall {h['recall']:.2f}, F1 {h['f1']:.2f}")
        print("Consider narrowing the sweep below the lowest threshold tested, or accept a small "
              "precision cost if the paraphrased recall gain is judged worth it — this is a "
              "judgment call to make explicitly in Chapter 8, not something to pick silently.")

    with open("crisis_cosine_similarity_detail.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "utterance", "max_similarity", "closest_tracked_phrase", "substring_detected"])
        for cat in ("direct", "paraphrased", "negative"):
            for u, s, phrase, hit in detail[cat]:
                writer.writerow([cat, u, f"{s:.4f}", phrase, hit])
    print("\nPer-utterance similarity scores + substring verdicts written to crisis_cosine_similarity_detail.csv")

    with open("crisis_cosine_threshold_sweep.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold",
                          "cosine_precision", "cosine_recall", "cosine_f1", "cosine_direct_recall", "cosine_paraphrased_recall",
                          "hybrid_precision", "hybrid_recall", "hybrid_f1", "hybrid_direct_recall", "hybrid_paraphrased_recall"])
        for r in results:
            c, h = r["cosine"], r["hybrid"]
            writer.writerow([r["threshold"],
                              f"{c['precision']:.4f}", f"{c['recall']:.4f}", f"{c['f1']:.4f}", f"{c['direct_recall']:.4f}", f"{c['paraphrased_recall']:.4f}",
                              f"{h['precision']:.4f}", f"{h['recall']:.4f}", f"{h['f1']:.4f}", f"{h['direct_recall']:.4f}", f"{h['paraphrased_recall']:.4f}"])
    print("Full threshold sweep (cosine + hybrid) written to crisis_cosine_threshold_sweep.csv — paste directly into Ch.7.")
    print(f"\n(Substring baseline for reference: Precision {baseline['precision']:.2f}, Recall {baseline['recall']:.2f}, "
          f"F1 {baseline['f1']:.2f}, Direct recall {baseline['direct_recall']:.2f}, Paraphrased recall {baseline['paraphrased_recall']:.2f})")


if __name__ == "__main__":
    main()
