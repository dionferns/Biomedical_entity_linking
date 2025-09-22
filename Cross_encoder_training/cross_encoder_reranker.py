import os
import json
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class CrossEncoderReranker(nn.Module):
    """Memory-efficient cross-encoder using separate encoding + lightweight similarity head.
    
    Instead of encoding all pairs together (which causes OOM), this encodes mentions
    and candidates separately, then computes similarity with a small MLP.
    """

    def __init__(
        self,
        model_name: str = "michiyasunaga/BioLinkBERT-base",
        use_definitions: bool = False,
        entities_path: Optional[str] = None,
        max_length: int = 128,
        hidden_size: int = 768,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.use_definitions = bool(use_definitions)
        self.max_length = int(max_length)
        
        # Lightweight similarity head: mention + candidate -> score
        self.similarity_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
        )
        self.nota_bias = nn.Parameter(torch.tensor(0.0))

        # Load entity names/definitions for candidate text
        self._cui2name: dict[str, str] = {}
        self._cui2def: dict[str, str] = {}
        if entities_path and os.path.exists(entities_path):
            try:
                import ijson
                with open(entities_path, "rb") as f:
                    for ent in ijson.items(f, "item"):
                        cui = ent.get("cui", "")
                        if not cui:
                            continue
                        self._cui2name[cui] = ent.get("name", "") or cui
                        if self.use_definitions:
                            self._cui2def[cui] = ent.get("definition", "") or ""
            except Exception:
                pass

    def _encode_texts(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """Encode texts separately and return [B, H] embeddings."""
        if not texts:
            return torch.empty((0, self.encoder.config.hidden_size), device=device)
        
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.no_grad():
            out = self.encoder(**batch).last_hidden_state[:, 0]  # [CLS]
        return out

    def _build_candidate_texts(self, cand_cuis: List[List[str]]) -> List[str]:
        """Convert CUI lists to candidate text strings."""
        texts = []
        for cuis in cand_cuis:
            for cui in cuis:
                name = self._cui2name.get(cui, cui)
                text = name.strip()
                if self.use_definitions:
                    defin = self._cui2def.get(cui, "").strip()
                    if defin:
                        text = f"{name}. {defin}" if text else defin
                if text and not text.endswith((".", "!", "?")):
                    text = text + "."
                texts.append(text)
        return texts

    def score_batch(
        self,
        mention_texts: List[str],
        cand_cuis: List[List[str]],
        device: torch.device,
        faiss_prior: Optional[torch.Tensor] = None,
        faiss_weight: float = 0.0,
    ) -> torch.Tensor:
        """Return logits shaped [N, K+1] (with NOTA column)."""
        if not mention_texts or not cand_cuis:
            return torch.zeros((len(cand_cuis), 1), device=device)

        # 1. Encode mentions separately
        mention_embs = self._encode_texts(mention_texts, device=device)  # [N, H]
        
        # 2. Encode candidates separately (flattened)
        candidate_texts = self._build_candidate_texts(cand_cuis)
        candidate_embs = self._encode_texts(candidate_texts, device=device)  # [sum(K_i), H]
        
        # 3. Compute similarity scores per mention-candidate pair
        scores_list = []
        offset = 0
        for cuis in cand_cuis:
            if not cuis:
                scores_list.append(torch.empty((0,), device=device))
                continue
            
            k = len(cuis)
            # Get embeddings for this mention's candidates
            cand_embs = candidate_embs[offset:offset + k]  # [K, H]
            mention_emb = mention_embs[len(scores_list):len(scores_list) + 1]  # [1, H]
            
            # Concatenate mention + candidate embeddings
            combined = torch.cat([
                mention_emb.expand(k, -1),  # [K, H]
                cand_embs,                   # [K, H]
            ], dim=1)  # [K, 2H]
            
            # Score via similarity head
            pair_scores = self.similarity_head(combined).squeeze(-1)  # [K]
            scores_list.append(pair_scores)
            offset += k
        
        # Pad to max K
        max_k = max(len(s) for s in scores_list) if scores_list else 0
        if max_k == 0:
            scores = torch.zeros((len(scores_list), 0), device=device)
        else:
            scores = torch.zeros((len(scores_list), max_k), device=device)
            for i, s in enumerate(scores_list):
                if s.numel() > 0:
                    scores[i, :s.numel()] = s

        # Optional prior blend
        if faiss_prior is not None and faiss_weight != 0.0 and faiss_prior.numel() > 0:
            k_use = min(scores.size(1), faiss_prior.size(1))
            if k_use > 0:
                scores[:, :k_use] = scores[:, :k_use] + float(faiss_weight) * faiss_prior[:, :k_use].to(scores.dtype).to(scores.device)

        # Append NOTA column
        nota_col = self.nota_bias.view(1, 1).expand(scores.size(0), 1)
        logits = torch.cat([scores, nota_col], dim=1)
        return logits

    def score_and_loss(
        self,
        mention_texts: List[str],
        cand_cuis: List[List[str]],
        gold_indices: Optional[torch.LongTensor],
        device: torch.device,
        faiss_prior: Optional[torch.Tensor] = None,
        faiss_weight: float = 0.0,
        nota_weight: Optional[float] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Compute logits and (optional) loss."""
        logits = self.score_batch(
            mention_texts=mention_texts,
            cand_cuis=cand_cuis,
            device=device,
            faiss_prior=faiss_prior,
            faiss_weight=faiss_weight,
        )
        
        loss = None
        if gold_indices is not None:
            if nota_weight is None:
                loss = F.cross_entropy(logits, gold_indices)
            else:
                ce = F.cross_entropy(logits, gold_indices, reduction='none')
                nota_idx = logits.size(1) - 1
                w = torch.where(
                    gold_indices == nota_idx,
                    ce.new_full(gold_indices.shape, float(nota_weight)),
                    ce.new_ones(gold_indices.shape),
                )
                loss = (w * ce).mean()
        return loss, logits