"""
Training-optimized FAISS builder (augmented variant, synonyms-enabled, full-only).

Builds/loads FAISS artifacts from the augmented synonyms dictionaries created by
`improved_less_synonyms/run_dict_creators_aug2.py` (i.e., `ent_def_aug2_full_syn.json`), which
contain SNOMEDCT_US + MSH entities with ALL English, non‑suppressed synonyms
from MRCONSO, plus MedMention gold CUIs that were missing in the base set
expanded with their synonyms (when available under the same filters).

Notes:
- Full-only: no debug mode artifacts are produced by this builder.
- Names-only embeddings; definitions are not encoded into FAISS.
"""

import os
import json
import numpy as np
import torch
import faiss
import ijson
from transformers import BertModel, BertTokenizerFast
from typing import Optional


class TrainingOptimizedFaissBuilderAug2:
    def __init__(self, encoder: Optional[BertModel] = None, encoder_name: Optional[str] = None, use_gpu: Optional[bool] = None, gpu_device: int = 0):
        # Full-only mode
        self.debug = False
        self.debug_suffix = "full"
        self.training_mode = True
        can_use_cuda = torch.cuda.is_available()
        want_gpu = (use_gpu is True) or (use_gpu is None and can_use_cuda)
        self.device = torch.device(f"cuda:{gpu_device}" if want_gpu and can_use_cuda else "cpu")
        self.gpu_device = gpu_device

        model_name = encoder_name or "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        self.encoder = encoder or BertModel.from_pretrained(
            model_name,
        )
        self.encoder.to(self.device)
        self.tokenizer = BertTokenizerFast.from_pretrained(
            model_name,
        )
        self.encoder.eval()

        base_dir = "improved_less_synonyms/processed/entity_disambiguation"
        os.makedirs(base_dir, exist_ok=True)
        # Synonyms-augmented artifacts (full-only)
        self.entities_path = os.path.join(base_dir, "ent_def_aug2_full_syn.json")
        # Persist names-only embeddings/index permanently with a _name suffix
        self.embeddings_path = os.path.join(base_dir, "ent_embs_aug2_name_full.npy")
        self.faiss_index_path = os.path.join(base_dir, "faiss_index_aug2_name_full.bin")
        self.cuids_path = os.path.join(base_dir, "cuids_aug2_name_full.json")

        self.index = None
        self.cuids = []

        if os.path.exists(self.faiss_index_path):
            self.load_index()
        elif os.path.exists(self.embeddings_path) and os.path.exists(self.entities_path):
            self._build_index_from_existing_embeddings()
        elif os.path.exists(self.entities_path):
            self.build_index()

    def _encode_texts(self, texts):
        toks = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.encoder(**toks).last_hidden_state  # [B, T, H]
            mask = toks["attention_mask"].unsqueeze(-1).float()  # [B, T, 1]
            masked_sum = (outputs * mask).sum(dim=1)  # [B, H]
            lengths = mask.sum(dim=1).clamp(min=1.0)  # [B, 1]
            out = (masked_sum / lengths).cpu().numpy().astype("float32")  # masked mean pooling
        return out

    def build_index(self):
        if not os.path.exists(self.entities_path):
            print(f"Synonyms-augmented entities not found at {self.entities_path}")
            return

        # Batch size for encoding (full-only)
        encode_bs = 128

        # First pass: count entities to preallocate a memmapped embeddings file
        total_entities = 0
        with open(self.entities_path, "rb") as f:
            for _ in ijson.items(f, "item"):
                total_entities += 1

        if total_entities == 0:
            print("No entities found to build synonyms-augmented FAISS index")
            return

        # Second pass: stream entities, encode in batches (names-only), normalize, write to
        # memmapped .npy on disk, and add to FAISS index incrementally (GPU if available)
        cuids: list = []
        write_pos = 0
        dim = None
        self.index = None
        embs_mm = None  # np.memmap handle for embeddings .npy

        # Require GPU for FAISS building
        use_gpu_build = (self.device.type == "cuda") and hasattr(faiss, "StandardGpuResources")
        if not use_gpu_build:
            print("ERROR: CUDA is required for FAISS building. Please run on a GPU-enabled machine.")
            return
        cpu_index = None
        gpu_index = None
        gpu_res = None

        with open(self.entities_path, "rb") as f:
            parser = ijson.items(f, "item")
            batch_texts = []
            batch_sizes = []  # track batch sizes for sanity if needed

            for ent in parser:
                cuids.append(ent.get("cui", ""))
                # Encode names only (definitions are intentionally excluded)
                text = (ent.get("name", "") or "").strip()
                batch_texts.append(text)

                if len(batch_texts) == encode_bs:
                    embs = self._encode_texts(batch_texts)  # (B, H) float32
                    # L2-normalize for cosine similarity
                    norms = np.linalg.norm(embs, axis=1, keepdims=True)
                    norms = np.maximum(norms, 1e-12)
                    embs = embs / norms

                    if dim is None:
                        dim = int(embs.shape[1])
                        # Create memmapped .npy to store all embeddings without peak RAM
                        embs_mm = np.lib.format.open_memmap(
                            self.embeddings_path, mode="w+", dtype="float32", shape=(total_entities, dim)
                        )
                        # Initialize FAISS index (GPU or CPU)
                        cpu_index = faiss.IndexFlatIP(dim)
                        if use_gpu_build:
                            gpu_res = faiss.StandardGpuResources()
                            gpu_res.setTempMemory(512 * 1024 * 1024)
                            co = faiss.GpuClonerOptions()
                            co.useFloat16 = True
                            co.usePrecomputed = False
                            gpu_index = faiss.index_cpu_to_gpu(gpu_res, self.gpu_device, cpu_index, co)

                    # Write chunk to memmap and add to index
                    bsz = embs.shape[0]
                    embs_mm[write_pos:write_pos + bsz] = embs.astype("float32", copy=False)
                    if use_gpu_build and gpu_index is not None:
                        gpu_index.add(embs.astype("float32", copy=False))
                    else:
                        cpu_index.add(embs.astype("float32", copy=False))
                    write_pos += bsz
                    batch_sizes.append(bsz)
                    batch_texts.clear()

            # Flush remainder
            if batch_texts:
                embs = self._encode_texts(batch_texts)
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-12)
                embs = embs / norms
                if dim is None:
                    dim = int(embs.shape[1])
                    embs_mm = np.lib.format.open_memmap(
                        self.embeddings_path, mode="w+", dtype="float32", shape=(total_entities, dim)
                    )
                    cpu_index = faiss.IndexFlatIP(dim)
                    if use_gpu_build:
                        gpu_res = faiss.StandardGpuResources()
                        gpu_res.setTempMemory(512 * 1024 * 1024)  # 512MB scratch cap

                        co = faiss.GpuClonerOptions()
                        co.useFloat16 = True
                        co.usePrecomputed = False
                        gpu_index = faiss.index_cpu_to_gpu(gpu_res, self.gpu_device, cpu_index, co)
                bsz = embs.shape[0]
                embs_mm[write_pos:write_pos + bsz] = embs.astype("float32", copy=False)
                if use_gpu_build and gpu_index is not None:
                    gpu_index.add(embs.astype("float32", copy=False))
                else:
                    cpu_index.add(embs.astype("float32", copy=False))
                write_pos += bsz

        # Sanity: ensure we wrote expected number of rows
        if write_pos != total_entities:
            print(f"Warning: wrote {write_pos} embeddings, expected {total_entities}")

        # Finalize FAISS index: move GPU index back to CPU if needed
        if use_gpu_build and gpu_index is not None:
            # right before: self.index = faiss.index_gpu_to_cpu(gpu_index)
            try:
                del self.encoder
                del self.tokenizer
            except Exception:
                pass
            torch.cuda.empty_cache()

            self.index = faiss.index_gpu_to_cpu(gpu_index)
            # Cleanup GPU objects
            del gpu_index
            if gpu_res is not None:
                del gpu_res
        else:
            self.index = cpu_index

        # Persist CUIDs and FAISS index
        self.cuids = cuids
        with open(self.cuids_path, "w", encoding="utf-8") as f:
            json.dump(self.cuids, f, ensure_ascii=False, indent=2)
        faiss.write_index(self.index, self.faiss_index_path)
        print(f"Built synonyms-augmented FAISS index: {self.index.ntotal} vectors")

    def _build_index_from_existing_embeddings(self):
        # Load embeddings lazily via memmap (read-only)
        embs = np.load(self.embeddings_path, mmap_mode="r")

        # Stream CUIs from entities file to avoid loading full JSON in memory
        cuids: list = []
        with open(self.entities_path, "rb") as f:
            for ent in ijson.items(f, "item"):
                cuids.append(ent.get("cui", ""))
        self.cuids = cuids
        with open(self.cuids_path, "w", encoding="utf-8") as f:
            json.dump(self.cuids, f, ensure_ascii=False, indent=2)

        # Build FAISS index by adding in chunks to bound RAM (GPU if available)
        dim = int(embs.shape[1])
        use_gpu_build = (self.device.type == "cuda") and hasattr(faiss, "StandardGpuResources")
        cpu_index = faiss.IndexFlatIP(dim)
        gpu_index = None
        gpu_res = None
        if use_gpu_build:
            gpu_res = faiss.StandardGpuResources()
            gpu_res.setTempMemory(512 * 1024 * 1024)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            co.usePrecomputed = False
            gpu_index = faiss.index_cpu_to_gpu(gpu_res, self.gpu_device, cpu_index, co)

        chunk_size = 100000
        for start in range(0, embs.shape[0], chunk_size):
            chunk = np.array(embs[start:start + chunk_size], dtype="float32", copy=False)
            norms = np.linalg.norm(chunk, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            chunk = chunk / norms
            if use_gpu_build and gpu_index is not None:
                gpu_index.add(chunk)
            else:
                cpu_index.add(chunk)

        try:
            del self.encoder
            del self.tokenizer
        except Exception:
            pass
        torch.cuda.empty_cache()

        self.index = faiss.index_gpu_to_cpu(gpu_index) if (use_gpu_build and gpu_index is not None) else cpu_index
        if gpu_index is not None:
            del gpu_index
        if gpu_res is not None:
            del gpu_res

        faiss.write_index(self.index, self.faiss_index_path)
        print(f"Rebuilt synonyms-augmented FAISS index: {self.index.ntotal} vectors")

    def load_index(self):
        if not os.path.exists(self.faiss_index_path):
            print(f"Synonyms-augmented FAISS index not found at {self.faiss_index_path}")
            return False
        self.index = faiss.read_index(self.faiss_index_path)
        if os.path.exists(self.cuids_path):
            with open(self.cuids_path, "r", encoding="utf-8") as f:
                self.cuids = json.load(f)
        print(f"Loaded synonyms-augmented FAISS index: {self.index.ntotal} vectors")
        return True

