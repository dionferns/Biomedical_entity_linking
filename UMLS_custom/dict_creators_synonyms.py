import os, json
import pyarrow.parquet as pq
from typing import Optional, List, Dict

from normalisation import normalise

class dict_creater_synonyms:
    def __init__(self,
                 restrict_sab: Optional[List[str]] = None,
                 restrict_def_sab: Optional[List[str]] = None):
        """Dictionary creator for synonyms-enabled entity lists (full-only).

        - Aliases/mentions: MRCONSO filtered to LAT=='ENG', SUPPRESS=='N', SAB∈restrict_sab.
          No ISPREF filter (include all English, non-suppressed strings), producing synonyms.
        - Definitions: MRDEF filtered to SUPPRESS=='N', SAB∈restrict_def_sab; first hit per CUI.
        - Full-only: assumes MRCONSO_full.parquet and MRDEF_full.parquet exist.
        """
        self.restrict_sab = restrict_sab or ["SNOMEDCT_US", "MSH"]
        self.restrict_def_sab = restrict_def_sab or ["NCI","MSH","HPO","CSP","MEDLINEPLUS","MTH"]

        self.save_path = "improved_less_synonyms/processed/entity_disambiguation"
        os.makedirs(self.save_path, exist_ok=True)

        # Full-only parquet inputs (must exist; we do not create them here)
        # Use the same filenames produced by base dict_creater conversions.
        self.mrconso_pq_path = os.path.join("data/raw/UMLS", "MRCONSO_pq_full.parquet")
        self.mrdef_pq_path = os.path.join("data/raw/UMLS", "MRDEF_pq_full.parquet")

        # Outputs (full-only)
        self.cui2def_path = os.path.join(self.save_path, "cui2def_full.json")
        self.ent_def_syn_full_path = os.path.join(self.save_path, "ent_def_syn_full.json")
        self.ent_def_aug2_full_syn_path = os.path.join(self.save_path, "ent_def_aug2_full_syn.json")
        self.cui2strdef_aug2_full_path = os.path.join(self.save_path, "cui2strdef_aug2_full.json")
        # Optional compatibility filename used by existing components
        self.cui2strdef_aug_full_compat_path = os.path.join(self.save_path, "cui2strdef_aug_full.json")

    def build_cui2def_full(self) -> None:
        df = pq.read_table(
            self.mrdef_pq_path,
            columns=['CUI','SAB','DEF','SUPPRESS']
        ).to_pandas()

        df = df[df.SUPPRESS == 'N']
        if self.restrict_def_sab:
            df = df[df.SAB.isin(self.restrict_def_sab)]
        df = df.drop_duplicates(subset=['CUI'], keep='first')

        cui2def_dict = dict(zip(df.CUI.astype(str), df.DEF.astype(str)))
        os.makedirs(os.path.dirname(self.cui2def_path), exist_ok=True)
        with open(self.cui2def_path, 'w', encoding='utf-8') as f:
            json.dump(cui2def_dict, f, ensure_ascii=False, indent=2)

    def build_ent_def_syn_full(self) -> None:
        """Build synonyms-expanded entity list from MRCONSO with filters (ENG, SUPPRESS=N, SAB∈restrict_sab).
        Includes all strings (no ISPREF filter), which yields multiple entries per CUI (synonyms).
        Output rows: {"cui", "name", "definition"} (definition filled later if needed).
        """
        df = pq.read_table(
            self.mrconso_pq_path,
            columns=['CUI','LAT','SAB','STR','SUPPRESS', 'ISPREF', 'TTY']
        ).to_pandas()

        df = df[(df.LAT == 'ENG') & (df.SUPPRESS == 'N')]
        if self.restrict_sab:
            df = df[df.SAB.isin(self.restrict_sab)]

        df = df[
            (df.ISPREF == 'Y') |
            ((df.SAB == 'SNOMEDCT_US') & (df.TTY.isin(['PT', 'FN']))) |
            ((df.SAB == 'MSH') & (df.TTY == 'MH'))
        ]
        df['STR_norm'] = df['STR'].astype(str).apply(normalise)
        df = df.drop_duplicates(subset=['CUI','STR_norm'], keep='first')
        # normalise the data first. 
        # We deliberately do NOT drop duplicates on CUI; keep synonyms (but avoid exact dup (CUI, STR))
        # df = df.drop_duplicates(subset=['CUI','STR'], keep='first')

        # Prepare basic entries; definition to be attached by join in augmentation step if desired
        entities = [
            {"cui": str(row.CUI), "name": str(row.STR_norm), "definition": ""}
            for _, row in df.iterrows()
        ]

        with open(self.ent_def_syn_full_path, 'w', encoding='utf-8') as f:
            json.dump(entities, f, ensure_ascii=False, indent=2)

    def augment_with_medmention_missing(self, debug: bool = False) -> None:
        """Augment the synonyms entity list with MedMention gold CUIs missing from base, adding their synonyms
        from MRCONSO under the same filters when present; if absent, add a single entry using gold mention.
        """
        # Ensure prerequisites
        if not os.path.isfile(self.ent_def_syn_full_path):
            raise FileNotFoundError("ent_def_syn_full.json not found; run build_ent_def_syn_full() first.")
        if not os.path.isfile(self.cui2def_path):
            raise FileNotFoundError("cui2def_full.json not found; run build_cui2def_full() first.")

        with open(self.ent_def_syn_full_path, 'r', encoding='utf-8') as f:
            base_entities = json.load(f)
        with open(self.cui2def_path, 'r', encoding='utf-8') as f:
            cui2def = json.load(f)

        base_cuis = {e.get('cui', '') for e in base_entities}

        # Load MedMention mapping to collect gold CUIs and example mentions (full-only uses full splits)
        from data_processing.mention_detection.md_preprocessing_v2 import MentionDetectionPreprocessingV2
        splits = ["train", "val", "test"]
        gold_cuis: set[str] = set()
        cui_to_mention: Dict[str, str] = {}
        for split in splits:
            md = MentionDetectionPreprocessingV2(debug=debug, split_type=split)
            md.create_train_val_test_splits()
            if not (os.path.exists(md.abstract_dict_filepath) and os.path.exists(md.spans_dict_filepath)):
                md.parse_medmention_to_json()
            if not os.path.exists(md.id2cuistr_filepath):
                md.build_id2cuistr()
            import json as _json
            with open(md.id2cuistr_filepath, 'r', encoding='utf-8') as f:
                id2cuistr = _json.load(f)
            for rec in id2cuistr.values():
                cui = (rec.get('cui', '') or '').strip()
                mention = (rec.get('mention', '') or '').strip()
                if cui:
                    gold_cuis.add(cui)
                    if mention and cui not in cui_to_mention:
                        cui_to_mention[cui] = mention

        missing_cuis = [c for c in gold_cuis if c and c not in base_cuis]
        print(f"Missing MedMention CUIs not in base (synonyms list): {len(missing_cuis)}")

        # Fetch synonyms for missing CUIs from MRCONSO under same filters
        dfc = pq.read_table(
            self.mrconso_pq_path,
            columns=['CUI','LAT','SAB','STR','SUPPRESS', 'ISPREF', 'TTY']
        ).to_pandas()
        dfc = dfc[(dfc.LAT == 'ENG') & (dfc.SUPPRESS == 'N')]
        if self.restrict_sab:
            dfc = dfc[dfc.SAB.isin(self.restrict_sab)]

        dfc = dfc[
            (dfc.ISPREF == 'Y') |
            ((dfc.SAB == 'SNOMEDCT_US') & (dfc.TTY.isin(['PT', 'FN']))) |
            ((dfc.SAB == 'MSH') & (dfc.TTY == 'MH'))
        ]

        dfc['STR_norm'] = dfc['STR'].astype(str).apply(normalise)
        dfc = dfc.drop_duplicates(subset=['CUI','STR_norm'], keep='first')

        dfc['CUI'] = dfc['CUI'].astype(str)

        augmented = list(base_entities)
        for cui in missing_cuis:
            syn_rows = dfc[dfc.CUI == cui]
            if not syn_rows.empty:
                # Add all synonyms for this CUI
                for _, row in syn_rows.drop_duplicates(subset=['CUI','STR_norm'], keep='first').iterrows():
                    augmented.append({
                        "cui": cui,
                        "name": str(row.STR_norm),
                        "definition": cui2def.get(cui, "")
                    })
            else:
                # No synonyms available under the filters; add one entry using example mention if present
                name = normalise(cui_to_mention.get(cui, cui))
                augmented.append({
                    "cui": cui,
                    "name": name,
                    "definition": cui2def.get(cui, "")
                })

        # Write ent_def_aug2_full_syn.json
        with open(self.ent_def_aug2_full_syn_path, 'w', encoding='utf-8') as f:
            json.dump(augmented, f, ensure_ascii=False, indent=2)

        # Also write cui2strdef_aug2_full.json for convenience
        cui2strdef_aug = {
            e.get('cui', ''): {
                "name": e.get('name', ''),
                "definition": e.get('definition', '')
            }
            for e in augmented if e.get('cui')
        }
        with open(self.cui2strdef_aug2_full_path, 'w', encoding='utf-8') as f:
            json.dump(cui2strdef_aug, f, ensure_ascii=False, indent=2)
        # Also write a compatibility copy for components expecting cui2strdef_aug_full.json
        try:
            with open(self.cui2strdef_aug_full_compat_path, 'w', encoding='utf-8') as f:
                json.dump(cui2strdef_aug, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

