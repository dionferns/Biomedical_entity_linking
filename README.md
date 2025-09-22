## Biomedical Entity Linking Pipeline

A complete, modular pipeline for biomedical entity linking (EL) built around a mention detection component, custom UMLS resources, a bi-encoder retriever, and a cross-encoder re-ranker, ending with an end-to-end evaluation.

The recommended order of operations:

1) Train mention detection → `Mention_detection/train_md_only.py`
2) Build custom UMLS resources (dictionaries, FAISS index) → `UMLS_custom/`
3) Fine-tune bi-encoder retriever → `bi_encoder_training/`
4) Train cross-encoder re-ranker → `cross_encoder_training/`
5) Combine trained components and run end-to-end test → `full_pipeline/full_el_test.py`


## Repository structure

- `Mention_detection/`: mention detection preprocessing, training, testing
- `UMLS_custom/`: custom UMLS synonym dictionaries and FAISS index creation
- `bi_encoder_training/`: bi-encoder contrastive training and FAISS verification
- `cross_encoder_training/`: context building, candidate retrieval, CE data combining, CE training, CE evaluation
- `full_pipeline/`: end-to-end EL evaluation script
- `data/`: raw and processed datasets (expected locations)


## Environment setup

- Python ≥ 3.9
- Recommended: a CUDA-enabled GPU for training and FAISS GPU

Install core dependencies (examples):

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\\Scripts\\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # choose CUDA/CPU build accordingly
pip install transformers sentencepiece
pip install faiss-gpu  # or faiss-cpu if no GPU
pip install numpy matplotlib ijson
```

If you maintain a `requirements.txt`, you can use:

```bash
pip install -r requirements.txt
```


## Data layout

By default, scripts expect files under `data/` and `improved_less_synonyms/` (or specify paths via CLI flags):

- Mention detection (per split):
  - `data/processed/mention_detection/<split>/abstract_dict_<debug|full>.json`
  - `data/processed/mention_detection/<split>/spans_dict_<debug|full>.json`
  - `data/processed/mention_detection/<split>/id2cuistr_dict_<debug|full>.json`
  - `data/processed/mention_detection/<split>/tokenized_<split>_<debug|full>.pt`
- UMLS entity resources (examples):
  - `improved_less_synonyms/processed/entity_disambiguation/ent_def_aug2_full_syn.json`
  - `improved_less_synonyms/processed/entity_disambiguation/faiss_index_aug2_name_full.bin`
  - `improved_less_synonyms/processed/entity_disambiguation/cuids_aug2_name_full.json`


## Step 1 — Mention detection

Train the mention detector first to produce tokenized tensors, span annotations, and ID mappings used by later stages.

Typical workflow:

```bash
# (Optional) Preprocess/prepare MD inputs if needed
python Mention_detection/run_md_preprocessing_v2.py \
  --debug           # or omit for full

# Train mention detection
python Mention_detection/train_md_only.py \
  --debug           # or custom flags your script supports

# (Optional) Evaluate MD
python Mention_detection/test_md_only.py \
  --debug
```

Artifacts used later will be written under `data/processed/mention_detection/<split>/`.


## Step 2 — Build custom UMLS resources

Create dictionaries and build a FAISS index for entity candidate retrieval.

```bash
# Build dictionaries (synonyms, metadata, etc.)
python UMLS_custom/run_dict_creators_aug2.py \
  --output_dir improved_less_synonyms/processed/entity_disambiguation

# Build FAISS index over entity name variants
python UMLS_custom/run_faiss_builder_aug2.py \
  --base_dir improved_less_synonyms/processed/entity_disambiguation
```

Expected outputs include `faiss_index_*.bin`, `cuids_*.json`, and `ent_def_*.json` under the given directory.


## Step 3 — Bi-encoder fine-tuning

Fine-tune a bi-encoder (e.g., SapBERT/BioLinkBERT) for retrieval.

```bash
# (If needed) Build bi-encoder training data
python bi_encoder_training/build_bi_data2.py \
  --split train  # adjust flags to your data flow

# Train bi-encoder (contrastive)
python bi_encoder_training/train_biencoder_contrastive2.py \
  --output_dir bi_encoder_training/runs/bi_run_001

# (Optional) Evaluate FAISS retrieval quality
python bi_encoder_training/eval_faiss_retrieval_only2.py \
  --index_path improved_less_synonyms/processed/entity_disambiguation/faiss_index_aug2_name_full.bin \
  --cuids_path improved_less_synonyms/processed/entity_disambiguation/cuids_aug2_name_full.json
```

Note: Cross-encoder candidate retrieval can use either pretrained encoders or your fine-tuned bi-encoder.


## Step 4 — Cross-encoder training

Prepare contexts, generate candidates, combine into training groups, and train the cross-encoder.

1) Build mention-centered contexts (with `[M_START]`/`[M_END]` markers):

```bash
python cross_encoder_training/build_ce_contexts.py \
  --abstract_path data/processed/mention_detection/val/abstract_dict_full.json \
  --spans_path    data/processed/mention_detection/val/spans_dict_full.json \
  --output_path   cross_encoder/ce_val_mention_sentences.jsonl.gz \
  --window 256
```

2) Generate candidates via FAISS for each mention (K per mention):

```bash
python cross_encoder_training/eval_faiss_candidates_dump.py \
  --split val \
  --K 10 \
  --tokenized_path data/processed/mention_detection/val/tokenized_val_full.pt \
  --id2cuistr_path data/processed/mention_detection/val/id2cuistr_dict_full.json \
  --ent_def_path improved_less_synonyms/processed/entity_disambiguation/ent_def_aug2_full_syn.json \
  --index_path   improved_less_synonyms/processed/entity_disambiguation/faiss_index_aug2_name_full.bin \
  --cuids_path   improved_less_synonyms/processed/entity_disambiguation/cuids_aug2_name_full.json \
  --candidates_json_out cross_encoder/candidate_data_val.json
```

3) Combine contexts + candidates into per-candidate JSONL.GZ suitable for CE training:

```bash
python cross_encoder_training/combine_ce_eval_and_contexts.py \
  --eval_jsonl_gz cross_encoder/candidates_val_J10_full.jsonl.gz \
  --contexts_jsonl_gz cross_encoder/ce_val_mention_sentences.jsonl.gz \
  --group_prefix val_full \
  --mrdef_path cross_encoder/MRDEF_unique_definitions.RRF \
  --def_truncation chars --def_max_chars 100 \
  --out_jsonl_gz cross_encoder/combined_val_full.jsonl.gz
```

4) Train the cross-encoder on the combined files:

```bash
python cross_encoder_training/train_ce_from_combined.py \
  --train_jsonl_gz cross_encoder/combined_train_full.jsonl.gz \
  --eval_jsonl_gz  cross_encoder/combined_val_full.jsonl.gz \
  --output_dir     cross_encoder/ce_runs/ce_run_001 \
  --encoder_name   cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
  --max_length 128 --groups_per_batch 16 --epochs 6 --lr 2e-5
```

5) (Optional) Evaluate the trained CE on test split using the same pipeline:

```bash
python cross_encoder_training/evaluate_ce_on_test.py \
  --tokenized_path data/processed/mention_detection/test/tokenized_test_full.pt \
  --abstract_path  data/processed/mention_detection/test/abstract_dict_full.json \
  --spans_path     data/processed/mention_detection/test/spans_dict_full.json \
  --id2cuistr_path data/processed/mention_detection/test/id2cuistr_dict_full.json \
  --encoder_name   cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
  --K 10 \
  --model_dir cross_encoder/ce_runs/ce_run_001/best \
  --contexts_jsonl_gz   cross_encoder/ce_test_mention_sentences.jsonl.gz \
  --candidates_jsonl_gz cross_encoder/candidates_test_J10_full.jsonl.gz \
  --combined_jsonl_gz   cross_encoder/combined_test_full.jsonl.gz
```

Note: `evaluate_ce_on_test.py` reuses functions from other CE scripts. If running as a module, ensure the CE scripts are importable as a package (e.g., place them under `cross_encoder/` with an `__init__.py` or set `PYTHONPATH` appropriately).


## Step 5 — End-to-end evaluation

After training MD, building UMLS resources, and training the bi-encoder and cross-encoder, run the full pipeline:

```bash
python full_pipeline/full_el_test.py \
  --tokenized_path data/processed/mention_detection/test/tokenized_test_full.pt \
  --abstract_path  data/processed/mention_detection/test/abstract_dict_full.json \
  --spans_path     data/processed/mention_detection/test/spans_dict_full.json \
  --id2cuistr_path data/processed/mention_detection/test/id2cuistr_dict_full.json \
  --encoder_name   cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
  --K 10 \
  --model_dir cross_encoder/ce_runs/ce_run_001/best
```

This script orchestrates: building contexts → generating candidates → combining inputs → loading the cross-encoder checkpoint → computing metrics (acc@K, NIL accuracy, MRR, etc.).


## Outputs and artifacts

- Mention detection: processed dicts and tokenized tensors per split under `data/processed/mention_detection/`
- UMLS resources: FAISS index, CUIs list, entity definitions under `improved_less_synonyms/processed/entity_disambiguation/`
- Bi-encoder: run/checkpoint dirs under `bi_encoder_training/runs/`
- Cross-encoder: run dirs with checkpoints and `metrics.jsonl` under `cross_encoder/ce_runs/`
- Combined CE data: `cross_encoder/combined_<group>.jsonl.gz`
- Evaluation candidates: `cross_encoder/candidates_<split>_J{K}_<suffix>.jsonl.gz`


## Tips & troubleshooting

- Paths: All scripts accept explicit paths; defaults assume the layout above. Use absolute paths if running from different working directories.
- FAISS GPU: Enable `--use_faiss_gpu` for faster retrieval; consider `--faiss_fp16_index` to reduce GPU memory.
- Definitions: Provide `--mrdef_path` for candidate definitions (or leave empty to skip adding definitions).
- Imports: Some CE scripts import others as `cross_encoder.*`. If you prefer running them directly from this folder, set `PYTHONPATH` to include `cross_encoder_training` as `cross_encoder`, or refactor into a `cross_encoder/` package.
- Data integrity: The CE context builder validates mention span alignment and will fail fast if spans don’t match abstract text.


## Citation

If you use this pipeline, please cite the relevant base models/datasets (e.g., SapBERT, BioLinkBERT, MedMentions, UMLS, FAISS) according to their licenses and papers.