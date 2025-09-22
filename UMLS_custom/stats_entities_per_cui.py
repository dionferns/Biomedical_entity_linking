import os
import argparse
from typing import Dict
from collections import Counter, defaultdict


def analyze_entities(entities_path: str, max_k_print: int = 20) -> None:
    """Stream the entities JSON (large) and report per-CUI counts distribution.

    Prints, for each k (number of texts per CUI) up to max_k_print:
      - number of CUIs with exactly k texts
      - percentage of CUIs
      - percentage of total texts contributed by those CUIs

    Remaining k > max_k_print are aggregated under a single bucket.
    """
    if not os.path.exists(entities_path):
        raise FileNotFoundError(f"Entities file not found: {entities_path}")

    try:
        import ijson  # streaming parser for large JSON arrays
    except Exception as e:
        raise RuntimeError(
            "ijson is required for streaming this large JSON file. Install with 'pip install ijson'."
        ) from e

    # Count texts per CUI
    per_cui_counts: Dict[str, int] = defaultdict(int)
    total_texts = 0

    with open(entities_path, "rb") as f:
        for ent in ijson.items(f, "item"):
            cui = (ent.get("cui", "") or "").strip()
            if not cui:
                continue
            per_cui_counts[cui] += 1
            total_texts += 1

    total_cuis = len(per_cui_counts)
    if total_cuis == 0:
        print("No CUIs found.")
        return

    # Histogram over k = texts per CUI
    hist = Counter(per_cui_counts.values())  # k -> number of CUIs with that k

    # Summary stats
    min_k = min(hist) if hist else 0
    max_k = max(hist) if hist else 0
    mean_k = total_texts / float(total_cuis)

    print(f"Entities path: {entities_path}")
    print(f"Total CUIs: {total_cuis}")
    print(f"Total texts: {total_texts}")
    print(f"Texts per CUI: min={min_k} mean={mean_k:.3f} max={max_k}")
    print("")
    print(f"Distribution by texts-per-CUI (k) up to k={max_k_print}:")
    print("k\tnum_cuis\t%_cuis\t%_texts")

    shown_texts = 0
    shown_cuis = 0
    for k in sorted(hist.keys()):
        if k > max_k_print:
            continue
        num_cuis_k = hist[k]
        pct_cuis = 100.0 * num_cuis_k / float(total_cuis)
        pct_texts = 100.0 * (num_cuis_k * k) / float(total_texts) if total_texts > 0 else 0.0
        shown_cuis += num_cuis_k
        shown_texts += num_cuis_k * k
        print(f"{k}\t{num_cuis_k}\t{pct_cuis:.2f}\t{pct_texts:.2f}")

    # Aggregate the tail
    tail_cuis = total_cuis - shown_cuis
    tail_texts = total_texts - shown_texts
    if tail_cuis > 0:
        pct_cuis_tail = 100.0 * tail_cuis / float(total_cuis)
        pct_texts_tail = 100.0 * tail_texts / float(total_texts) if total_texts > 0 else 0.0
        print(f">{max_k_print}\t{tail_cuis}\t{pct_cuis_tail:.2f}\t{pct_texts_tail:.2f}")


def main():
    ap = argparse.ArgumentParser(description="Report distribution of number of texts per CUI from FAISS entities file.")
    ap.add_argument(
        "--entities_path",
        type=str,
        default="improved_less_synonyms/processed/entity_disambiguation/ent_def_aug2_full_syn.json",
    )
    ap.add_argument("--max_k_print", type=int, default=20, help="Show buckets up to this k; aggregate the rest")
    args = ap.parse_args()

    analyze_entities(args.entities_path, int(args.max_k_print))


if __name__ == "__main__":
    main()


