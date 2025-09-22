import os
import argparse
from improved_less_synonyms.faiss_builder_training_aug2 import TrainingOptimizedFaissBuilderAug2


def main():
    ap = argparse.ArgumentParser(description="Build/Load synonyms-augmented FAISS index")
    ap.add_argument("--encoder_name", type=str, default="",
                    help="HF model id or local path for the encoder/tokenizer")
    args = ap.parse_args()

    # Ensure entities exist.
    base_dir = "improved_less_synonyms/processed/entity_disambiguation"
    entities = os.path.join(base_dir, "ent_def_aug2_full_syn.json")
    if not os.path.isfile(entities):
        raise FileNotFoundError(
            f"Missing {entities}. Run improved_less_synonyms/run_dict_creators_aug2.py first."
        )

    builder = TrainingOptimizedFaissBuilderAug2(encoder=None, encoder_name=args.encoder_name)
    if builder.index is None:
        raise RuntimeError("Failed to build or load the synonyms-augmented FAISS index.")

    print(f"Index vectors: {builder.index.ntotal}")
    print(f"Artifacts:\n  index: {builder.faiss_index_path}\n  cuids: {builder.cuids_path}\n  embs:  {builder.embeddings_path}")


if __name__ == "__main__":
    main()