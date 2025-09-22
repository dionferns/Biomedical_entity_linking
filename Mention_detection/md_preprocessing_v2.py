# from sklearn.model_selection import train_test_split
# from torch.utils.data import TensorDataset, DataLoader
# from collections import defaultdict
# import json
# import torch
# import os

# class MentionDetectionPreprocessingV2:
#     def __init__(self, debug, split_type):
#         """
#         Args:
#             debug: Whether to use debug or full data
#             split_type: One of 'train', 'val', 'test'
#         """
#         self.debug = debug
#         self.split_type = split_type  # 'train', 'val', or 'test'
#         self.debug_suffix = "debug" if self.debug else "full"
        
#         # Raw data path
#         self.filepath = f"data/raw/mention_detection/corpus_pubtator_{self.debug_suffix}.txt"
        
#         # Base processed directory
#         processed_dir = "data/processed/mention_detection"
        
#         # Create split-specific directories
#         self.split_dir = os.path.join(processed_dir, self.split_type)
#         os.makedirs(self.split_dir, exist_ok=True)
        
#         # Paths for raw split files
#         self.raw_splits_dir = os.path.join(processed_dir, "raw_splits")
#         os.makedirs(self.raw_splits_dir, exist_ok=True)
        
#         self.train_raw_file = os.path.join(self.raw_splits_dir, f"train_{self.debug_suffix}.txt")
#         self.val_raw_file = os.path.join(self.raw_splits_dir, f"val_{self.debug_suffix}.txt")
#         self.test_raw_file = os.path.join(self.raw_splits_dir, f"test_{self.debug_suffix}.txt")
        
#         # Paths for processed files specific to this split
#         self.abstract_dict_filepath = os.path.join(self.split_dir, f"abstract_dict_{self.debug_suffix}.json")
#         self.spans_dict_filepath = os.path.join(self.split_dir, f"spans_dict_{self.debug_suffix}.json")
#         self.id2cuistr_filepath = os.path.join(self.split_dir, f"id2cuistr_dict_{self.debug_suffix}.json")
#         self.tokenized_filepath = os.path.join(self.split_dir, f"tokenized_{self.split_type}_{self.debug_suffix}.pt")
    
#     def create_train_val_test_splits(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
#         """Split the raw MedMentions file into train, val, and test subsets."""
#         # Check if splits already exist
#         if (os.path.exists(self.train_raw_file) and 
#             os.path.exists(self.val_raw_file) and 
#             os.path.exists(self.test_raw_file)):
#             print(f"Splits already exist for {self.debug_suffix} data")
#             return
        
#         print(f"Creating train/val/test splits for {self.debug_suffix} data...")
        
#         with open(self.filepath, "r", encoding="utf-8") as f:
#             raw_text = f.read().strip()
        
#         # Documents are separated by blank lines
#         docs = raw_text.split("\n\n") if raw_text else []
        
#         # First split into train+val and test
#         train_val_docs, test_docs = train_test_split(
#             docs, test_size=test_ratio, random_state=42
#         )
        
#         # Then split train+val into train and val
#         adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
#         train_docs, val_docs = train_test_split(
#             train_val_docs, test_size=adjusted_val_ratio, random_state=42
#         )
        
#         # Save the splits
#         with open(self.train_raw_file, "w", encoding="utf-8") as f:
#             f.write("\n\n".join(train_docs))
        
#         with open(self.val_raw_file, "w", encoding="utf-8") as f:
#             f.write("\n\n".join(val_docs))
        
#         with open(self.test_raw_file, "w", encoding="utf-8") as f:
#             f.write("\n\n".join(test_docs))
        
#         print(f"Created splits: train={len(train_docs)}, val={len(val_docs)}, test={len(test_docs)}")
    
#     def get_split_file(self):
#         """Get the appropriate split file based on split_type"""
#         if self.split_type == "train":
#             return self.train_raw_file
#         elif self.split_type == "val":
#             return self.val_raw_file
#         elif self.split_type == "test":
#             return self.test_raw_file
#         else:
#             raise ValueError(f"Invalid split_type: {self.split_type}")
    
#     def parse_medmention_to_json(self):
#         """Parse the split-specific MedMention data to JSON"""
#         # Ensure splits exist
#         self.create_train_val_test_splits()
        
#         split_file = self.get_split_file()
        
#         print(f"Parsing {self.split_type} split for {self.debug_suffix} data...")
        
#         with open(split_file, encoding="utf-8") as f:
#             lines = [line.strip() for line in f if line.strip()]
        
#         text_dict = {}
#         raw_spans_dict = defaultdict(list)
#         span_counter = 0
        
#         for line in lines:
#             if "|t|" in line or "|a|" in line:
#                 pmid, tag, content = line.split("|", 2)
#                 content = content.strip()
#                 text_dict.setdefault(pmid, "")
#                 sep = " " if text_dict[pmid] else ""
#                 text_dict[pmid] += sep + content
#             else:
#                 parts = line.split("\t")
#                 if len(parts) >= 6:
#                     pmid = parts[0]
#                     start_char = int(parts[1])
#                     end_char = int(parts[2])
#                     mention = parts[3]
#                     cui = parts[5]
                    
#                     span_counter += 1
                    
#                     raw_spans_dict[pmid].append({
#                         "id": span_counter,
#                         "start": start_char,
#                         "end": end_char,
#                         "mention": mention,
#                         "cui": cui
#                     })
        
#         raw_spans = dict(raw_spans_dict)
        
#         # Save to split-specific directory
#         with open(self.abstract_dict_filepath, "w", encoding="utf-8") as f:
#             json.dump(text_dict, f, ensure_ascii=False, indent=2)
        
#         with open(self.spans_dict_filepath, "w", encoding="utf-8") as f:
#             json.dump(raw_spans, f, ensure_ascii=False, indent=2)
        
#         print(f"Saved {len(text_dict)} abstracts and {len(raw_spans)} span groups for {self.split_type}")
    
#     def build_id2cuistr(self):
#         """Build ID to CUI/mention mapping for this split"""
#         if not os.path.exists(self.spans_dict_filepath):
#             self.parse_medmention_to_json()
        
#         id_to_cui = {}
        
#         with open(self.spans_dict_filepath, "r", encoding="utf-8") as f:
#             spans_json_dict = json.load(f)
        
#         for doc_id, mentions in spans_json_dict.items():
#             for mention in mentions:
#                 id_to_cui[mention["id"]] = {
#                     'cui': mention['cui'],
#                     'mention': mention['mention']
#                 }
        
#         with open(self.id2cuistr_filepath, "w", encoding="utf-8") as f:
#             json.dump(id_to_cui, f, ensure_ascii=False, indent=2)
        
#         print(f"Built id2cuistr mapping with {len(id_to_cui)} entries for {self.split_type}")
    
#     def bio_labelling(self, tokenizer):
#         """Create BIO labels for this split"""
#         if not os.path.exists(self.abstract_dict_filepath):
#             self.parse_medmention_to_json()
        
#         with open(self.abstract_dict_filepath, "r", encoding="utf-8") as f:
#             text_dict = json.load(f)
        
#         with open(self.spans_dict_filepath, "r", encoding="utf-8") as f:
#             raw_spans_dict = json.load(f)
        
#         print(f"BIO labelling for {self.split_type}: {len(text_dict)} abstracts")
        
#         tokenized_set = defaultdict(list)
        
#         special_ids = {
#             tokenizer.cls_token_id,
#             tokenizer.sep_token_id,
#             tokenizer.pad_token_id
#         }
        
#         for pmid, text in text_dict.items():
#             spans = raw_spans_dict.get(pmid, [])
            
#             encoding = tokenizer(
#                 text,
#                 return_offsets_mapping=True,
#                 padding="max_length",
#                 truncation=True,
#                 max_length=512,
#             )
            
#             offsets = encoding.pop("offset_mapping")
#             input_ids = encoding["input_ids"]
            
#             labels = [0] * len(offsets)
#             id_tensor = [0] * len(offsets)
            
#             for span in spans:
#                 token_indices = [
#                     idx for idx, (s, e) in enumerate(offsets)
#                     if not (e <= span["start"] or s >= span["end"])
#                 ]
                
#                 if not token_indices:
#                     continue
                
#                 labels[token_indices[0]] = 1  # "B"
#                 id_tensor[token_indices[0]] = span["id"]
                
#                 for idx in token_indices[1:]:
#                     labels[idx] = 2  # "I"
#                     id_tensor[idx] = span["id"]
            
#             # Mask special tokens
#             labels = [
#                 (label if token_id not in special_ids else -100)
#                 for label, token_id in zip(labels, input_ids)
#             ]
            
#             id_tensor = [
#                 (cui if token_id not in special_ids else 0)
#                 for cui, token_id in zip(id_tensor, input_ids)
#             ]
            
#             tokenized_set[pmid].append({
#                 "input_ids": torch.tensor(input_ids),
#                 "attention_mask": torch.tensor(encoding["attention_mask"]),
#                 "labels": torch.tensor(labels),
#                 "id_tensor": torch.tensor(id_tensor),
#             })
        
#         # Save tokenized data
#         torch.save(dict(tokenized_set), self.tokenized_filepath)
#         print(f"Saved tokenized data to {self.tokenized_filepath}")
        
#         return dict(tokenized_set)
    
#     @staticmethod
#     def create_dataloader(raw_data, batch_size, shuffle=True):
#         """Create DataLoader from tokenized data"""
#         flat = []
#         for ex_list in raw_data.values():
#             flat.extend(ex_list)
        
#         if not flat:
#             raise ValueError("No data to create DataLoader")
        
#         all_input_ids = torch.stack([ex["input_ids"] for ex in flat])
#         all_attention = torch.stack([ex["attention_mask"] for ex in flat])
#         all_labels = torch.stack([ex["labels"] for ex in flat])
#         all_cuis = torch.stack([ex["id_tensor"] for ex in flat])
        
#         dataset = TensorDataset(all_input_ids, all_attention, all_labels, all_cuis)
#         loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
#         return loader



from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict
import json
import torch
import os
import sys
sys.path.append('/workspace')
from utils.error_handling import safe_file_read, safe_json_load, safe_json_save, safe_torch_save
from utils.reproducibility import get_generator, set_all_seeds
from config import get_config

class MentionDetectionPreprocessingV2:
    def __init__(self, debug, split_type):
        """
        Args:
            debug: Whether to use debug or full data
            split_type: One of 'train', 'val', 'test'
        """
        # Get configuration
        self.config = get_config(debug=debug)
        
        self.debug = debug
        self.split_type = split_type  # 'train', 'val', or 'test'
        self.debug_suffix = self.config.debug_suffix
        
        # Raw data path
        self.filepath = f"{self.config.raw_dir}/mention_detection/corpus_pubtator_{self.debug_suffix}.txt"
        
        # Processed directories with proper split folders
        processed_dir = os.path.join(self.config.processed_dir, "mention_detection")
        self.split_dir = os.path.join(processed_dir, self.split_type)
        os.makedirs(self.split_dir, exist_ok=True)
        
        # Split-specific raw files.
        self.train_raw_file = os.path.join(processed_dir, "raw_splits", f"train_{self.debug_suffix}.txt")
        self.val_raw_file = os.path.join(processed_dir, "raw_splits", f"val_{self.debug_suffix}.txt")
        self.test_raw_file = os.path.join(processed_dir, "raw_splits", f"test_{self.debug_suffix}.txt")
        
        # Paths for processed files specific to this split.
        self.abstract_dict_filepath = os.path.join(self.split_dir, f"abstract_dict_{self.debug_suffix}.json")
        self.spans_dict_filepath = os.path.join(self.split_dir, f"spans_dict_{self.debug_suffix}.json")
        self.id2cuistr_filepath = os.path.join(self.split_dir, f"id2cuistr_dict_{self.debug_suffix}.json")
        self.tokenized_filepath = os.path.join(self.split_dir, f"tokenized_{self.split_type}_{self.debug_suffix}.pt")
        
        os.makedirs(os.path.dirname(self.train_raw_file), exist_ok=True)
    
    def create_train_val_test_splits(self, train_ratio=None, val_ratio=None, test_ratio=None):
        """Split the raw MedMentions file into train, val, and test subsets."""
        # Use config values if not provided
        if train_ratio is None:
            train_ratio = self.config.train_ratio
        if val_ratio is None:
            val_ratio = self.config.val_ratio
        if test_ratio is None:
            test_ratio = self.config.test_ratio
            
        # Check if splits already exist
        if (os.path.exists(self.train_raw_file) and 
            os.path.exists(self.val_raw_file) and 
            os.path.exists(self.test_raw_file)):
            print(f"Splits already exist for {self.debug_suffix} data")
            return
        
        print(f"Creating train/val/test splits for {self.debug_suffix} data...")
        
        # Use safe file reading
        raw_text = safe_file_read(self.filepath)
        
        # Documents are separated by blank lines
        docs = raw_text.split("\n\n") if raw_text else []
        
        # Sort docs to ensure consistent ordering before split
        docs = sorted(docs)  # Ensure deterministic ordering
        
        # First split into train+val and test
        train_val_docs, test_docs = train_test_split(
            docs, test_size=test_ratio, random_state=42
        )
        
        # Then split train+val into train and val
        adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
        train_docs, val_docs = train_test_split(
            train_val_docs, test_size=adjusted_val_ratio, random_state=42
        )
        
        # Save the splits
        with open(self.train_raw_file, "w", encoding="utf-8") as f:
            f.write("\n\n".join(train_docs))
        
        with open(self.val_raw_file, "w", encoding="utf-8") as f:
            f.write("\n\n".join(val_docs))
        
        with open(self.test_raw_file, "w", encoding="utf-8") as f:
            f.write("\n\n".join(test_docs))
        
        print(f"Created splits: train={len(train_docs)}, val={len(val_docs)}, test={len(test_docs)}")
    
    def get_split_file(self):
        """Get the appropriate split file based on split_type"""
        if self.split_type == "train":
            return self.train_raw_file
        elif self.split_type == "val":
            return self.val_raw_file
        elif self.split_type == "test":
            return self.test_raw_file
        else:
            raise ValueError(f"Invalid split_type: {self.split_type}")
    
    def parse_medmention_to_json(self):
        """Parse the split-specific MedMention data to JSON"""
        # Ensure splits exist
        self.create_train_val_test_splits()
        
        split_file = self.get_split_file()
        
        print(f"Parsing {self.split_type} split for {self.debug_suffix} data...")
        
        # Use safe file reading
        raw_content = safe_file_read(split_file)
        lines = [line.strip() for line in raw_content.split('\n') if line.strip()]
        
        text_dict = {}
        raw_spans_dict = defaultdict(list)
        span_counter = 0
        
        for line in lines:
            if "|t|" in line or "|a|" in line:
                pmid, tag, content = line.split("|", 2)
                content = content.strip()
                text_dict.setdefault(pmid, "")
                sep = " " if text_dict[pmid] else ""
                text_dict[pmid] += sep + content
            else:
                parts = line.split("\t")
                if len(parts) >= 6:
                    pmid = parts[0]
                    start_char = int(parts[1])
                    end_char = int(parts[2])
                    mention = parts[3]
                    cui = parts[5]
                    
                    span_counter += 1
                    
                    raw_spans_dict[pmid].append({
                        "id": span_counter,
                        "start": start_char,
                        "end": end_char,
                        "mention": mention,
                        "cui": cui
                    })
        
        raw_spans = dict(raw_spans_dict)
        
        # Save to split-specific directory using safe methods
        safe_json_save(text_dict, self.abstract_dict_filepath)
        safe_json_save(raw_spans, self.spans_dict_filepath)
        
        print(f"Saved {len(text_dict)} abstracts and {len(raw_spans)} span groups for {self.split_type}")
    
    def build_id2cuistr(self):
        """Build ID to CUI/mention mapping for this split"""
        if not os.path.exists(self.spans_dict_filepath):
            self.parse_medmention_to_json()
        
        id_to_cui = {}
        
        # Use safe JSON loading
        spans_json_dict = safe_json_load(self.spans_dict_filepath)
        
        for doc_id, mentions in spans_json_dict.items():
            for mention in mentions:
                id_to_cui[mention["id"]] = {
                    'cui': mention['cui'],
                    'mention': mention['mention']
                }
        
        # Use safe JSON saving
        safe_json_save(id_to_cui, self.id2cuistr_filepath)
        
        print(f"Built id2cuistr mapping with {len(id_to_cui)} entries for {self.split_type}")
    
    def bio_labelling(self, tokenizer, batch_size=None):
        """Create BIO labels for this split with optional batch processing"""
        if batch_size is None:
            batch_size = self.config.bio_label_batch_size
            
        if not os.path.exists(self.abstract_dict_filepath):
            self.parse_medmention_to_json()
        
        # Use safe JSON loading
        text_dict = safe_json_load(self.abstract_dict_filepath)
        raw_spans_dict = safe_json_load(self.spans_dict_filepath)
        
        print(f"BIO labelling for {self.split_type}: {len(text_dict)} abstracts")
        
        tokenized_set = defaultdict(list)
        
        special_ids = {
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id
        }
        
        for pmid, text in text_dict.items():
            spans = raw_spans_dict.get(pmid, [])
            
            encoding = tokenizer(
                text,
                return_offsets_mapping=True,
                padding="max_length",
                truncation=True,
                max_length=self.config.max_seq_length,
            )
            
            offsets = encoding.pop("offset_mapping")
            input_ids = encoding["input_ids"]
            
            labels = [0] * len(offsets)
            id_tensor = [0] * len(offsets)
            
            for span in spans:
                token_indices = [
                    idx for idx, (s, e) in enumerate(offsets)
                    if not (e <= span["start"] or s >= span["end"])
                ]
                
                if not token_indices:
                    continue
                
                labels[token_indices[0]] = 1  # "B"
                id_tensor[token_indices[0]] = span["id"]
                
                for idx in token_indices[1:]:
                    labels[idx] = 2  # "I"
                    id_tensor[idx] = span["id"]
            
            # Mask special tokens
            labels = [
                (label if token_id not in special_ids else -100)
                for label, token_id in zip(labels, input_ids)
            ]
            
            id_tensor = [
                (cui if token_id not in special_ids else 0)
                for cui, token_id in zip(id_tensor, input_ids)
            ]
            
            tokenized_set[pmid].append({
                "input_ids": torch.tensor(input_ids, device='cpu'),
                "attention_mask": torch.tensor(encoding["attention_mask"], device='cpu'),
                "labels": torch.tensor(labels, device='cpu'),
                "id_tensor": torch.tensor(id_tensor, device='cpu'),
            })
        
        # Save tokenized data using safe method
        safe_torch_save(dict(tokenized_set), self.tokenized_filepath)
        print(f"Saved tokenized data to {self.tokenized_filepath}")
        
        return dict(tokenized_set)
    
    @staticmethod
    def create_dataloader(raw_data, batch_size, shuffle=True, num_workers=None, seed_worker=None):
        """Create optimized DataLoader from tokenized data with parallel loading and reproducibility"""
        # Get config for default values
        config = get_config()
        if num_workers is None:
            num_workers = config.num_workers
        
        # Set up reproducibility
        if seed_worker is None:
            seed_worker = set_all_seeds(config.random_seed)
        generator = get_generator(config.random_seed)
            
        flat = []
        for ex_list in raw_data.values():
            flat.extend(ex_list)
        
        if not flat:
            raise ValueError("No data to create DataLoader")
        
        all_input_ids = torch.stack([ex["input_ids"] for ex in flat]).cpu()
        all_attention = torch.stack([ex["attention_mask"] for ex in flat]).cpu()
        all_labels = torch.stack([ex["labels"] for ex in flat]).cpu()
        all_cuis = torch.stack([ex["id_tensor"] for ex in flat]).cpu()
        
        # Ensure all tensors are on CPU for DataLoader compatibility
        
        dataset = TensorDataset(all_input_ids, all_attention, all_labels, all_cuis)
        
        # Create optimized DataLoader with parallel loading and reproducibility
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers,  # Parallel data loading
            pin_memory=torch.cuda.is_available(),  # Faster GPU transfer
            persistent_workers=(num_workers > 0),  # Keep workers alive
            prefetch_factor=2 if num_workers > 0 else None,  # Prefetch batches
            worker_init_fn=seed_worker if num_workers > 0 else None,  # Seed workers
            generator=generator  # Use seeded generator for shuffling
        )
        return loader