import os, json, random, argparse
from typing import Dict, List, Tuple, Optional

try:
    from normalisation import normalise as _norm
except Exception:
    def _norm(s: str) -> str:
        return (s or "").strip().lower()


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_jsonl(path: str, rows: List[dict]) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _dedup_texts(texts: List[str]) -> List[str]:
    seen = set()
    out = []
    for t in texts:
        t = _norm(t)
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def build_pool(entities_path: str, medmention_train_path: Optional[str]) -> Dict[str, List[str]]:    
    if not os.path.exists(entities_path):
        raise FileNotFoundError(f"Entities file not found: {entities_path}")
    ents = _read_json(entities_path)
    by_cui: Dict[str, List[str]] = {}
    # From FAISS entities (synonyms/aliases)
    for e in ents:
        cui = (e.get("cui") or "").strip()
        name = _norm(e.get("name") or "")
        if cui and name:
            by_cui.setdefault(cui, []).append(name)

    # From MedMention TRAIN only
    if medmention_train_path:
        if not os.path.exists(medmention_train_path):
            raise FileNotFoundError(f"MedMention TRAIN id2cuistr not found: {medmention_train_path}")
        id2 = _read_json(medmention_train_path)
        for rec in id2.values():
            cui = (rec.get("cui") or "").strip()
            txt = _norm(rec.get("mention") or "")
            if cui and txt:
                by_cui.setdefault(cui, []).append(txt)

    # De-duplicate per CUI and drop singleton CUIs (need at least 2 texts for contrastive)
    for cui in list(by_cui.keys()):
        by_cui[cui] = _dedup_texts(by_cui[cui])
        if len(by_cui[cui]) < 2:
            del by_cui[cui]
    return by_cui


def split_per_cui(by_cui: Dict[str, List[str]], dev_ratio: float, seed: int
                  ) -> Tuple[List[dict], List[dict], Dict[str, int]]:
    rng = random.Random(seed)
    train_rows: List[dict] = []
    dev_rows: List[dict] = []
    counts: Dict[str, int] = {}
    for cui, texts in by_cui.items():
        if not texts:
            continue
        t = list(texts)
        rng.shuffle(t)
        n = len(t)
        dev_n = max(0, int(round(dev_ratio * n)))
        # Keep at least 1 in train if possible
        if n - dev_n < 1 and n >= 2:
            dev_n = n - 1
        dev = t[:dev_n]
        trn = t[dev_n:]
        for s in trn:
            train_rows.append({"cui": cui, "text": s})
        for s in dev:
            dev_rows.append({"cui": cui, "text": s})
        counts[cui] = n
    return train_rows, dev_rows, counts


def main():
    ap = argparse.ArgumentParser(description="Build bi-encoder training/dev JSONL from FAISS entities + MedMention TRAIN mentions.")
    ap.add_argument("--entities_path", type=str,
                    default="improved_less_synonyms/processed/entity_disambiguation/ent_def_aug2_full_syn.json")
    ap.add_argument("--medmention_train_path", type=str,
                    default="data/processed/mention_detection/train/id2cuistr_dict_full.json")
    ap.add_argument("--dev_ratio", type=float, default=0.10, help="Per-CUI dev fraction (e.g., 0.10 = 10%).")
    ap.add_argument("--no_dev", action="store_true", help="If set, do not create a dev split or dev.jsonl.")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out_dir", type=str, default="improved_less_synonyms/processed/bi_contrastive")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pool = build_pool(args.entities_path, args.medmention_train_path)
    # If dev is disabled or ratio <= 0, put everything into train and skip dev
    if args.no_dev or float(args.dev_ratio) <= 0.0:
        train_rows = []
        dev_rows = []
        counts = {}
        for cui, texts in pool.items():
            for s in texts:
                train_rows.append({"cui": cui, "text": s})
            counts[cui] = len(texts)
    else:
        train_rows, dev_rows, counts = split_per_cui(pool, float(args.dev_ratio), int(args.seed))

    train_path = os.path.join(args.out_dir, "train.jsonl")
    dev_path = os.path.join(args.out_dir, "dev.jsonl")
    meta_path = os.path.join(args.out_dir, "meta.json")

    _write_jsonl(train_path, train_rows)
    write_dev = (not args.no_dev) and (float(args.dev_ratio) > 0.0) and (len(dev_rows) > 0)
    if write_dev:
        _write_jsonl(dev_path, dev_rows)
    meta = {
        "entities_path": args.entities_path,
        "medmention_train_path": args.medmention_train_path,
        "dev_ratio": float(args.dev_ratio),
        "no_dev": bool(args.no_dev),
        "seed": int(args.seed),
        "num_cuis": len(counts),
        "num_train_rows": len(train_rows),
        "num_dev_rows": len(dev_rows) if write_dev else 0,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote: {train_path} ({len(train_rows)} rows)")
    if write_dev:
        print(f"Wrote: {dev_path} ({len(dev_rows)} rows)")
    else:
        print("Dev split disabled; no dev.jsonl written.")
    print(f"Wrote: {meta_path} (CUIs={len(counts)})")


if __name__ == "__main__":
    main()


