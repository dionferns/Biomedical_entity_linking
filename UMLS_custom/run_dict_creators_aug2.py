"""
Runner to create synonyms-expanded entity dictionaries and augmented versions (full-only):

Outputs under data/processed/entity_disambiguation:
  - ent_def_syn_full.json
  - ent_def_aug2_full_syn.json
  - cui2strdef_aug2_full.json
  - cui2def_full.json (reused/build if missing)

Requires: data/raw/UMLS/MRCONSO_full.parquet and MRDEF_full.parquet.
"""

import os
from improved_less_synonyms.dict_creators_synonyms import dict_creater_synonyms
# from improved_less_synonyms.dict_creators import dict_creater
from data_processing.entity_disambiguation.dict_creators import dict_creater


RESTRICT_SABS = ["SNOMEDCT_US", "MSH"]
DEF_SABS = ["NCI","MSH","HPO","CSP","MEDLINEPLUS","MTH"]


def main():
    # Ensure MRCONSO/MRDEF parquet conversions exist using the base helper (full mode).
    base_dc = dict_creater(debug=False, restrict_sab=RESTRICT_SABS, restrict_def_sab=DEF_SABS)
    if not os.path.isfile(base_dc.mrconso_pq_path):
        print("→ Converting MRCONSO (full) to parquet …")
        base_dc.mrconso_rrf2pq()
    else:
        print("✓ MRCONSO parquet exists")
    if not os.path.isfile(base_dc.mrdef_pq_path):
        print("→ Converting MRDEF (full) to parquet …")
        base_dc.mrdef_rrf22pq()
    else:
        print("✓ MRDEF parquet exists")

    dc = dict_creater_synonyms(restrict_sab=RESTRICT_SABS, restrict_def_sab=DEF_SABS)

    # Build/verify definitions map
    if not os.path.isfile(dc.cui2def_path):
        print("→ Creating cui2def_full.json …")
        dc.build_cui2def_full()
    else:
        print("✓ cui2def_full.json already exists")

    # Build base synonyms list from MRCONSO (ENG, SUPPRESS=N, SAB∈restrict)
    if not os.path.isfile(dc.ent_def_syn_full_path):
        print("→ Creating ent_def_syn_full.json (synonyms list) …")
        dc.build_ent_def_syn_full()
    else:
        print("✓ ent_def_syn_full.json already exists")

    # Augment with MedMention-missing CUIs, pulling synonyms when available.
    print("→ Creating ent_def_aug2_full_syn.json and cui2strdef_aug2_full.json …")
    dc.augment_with_medmention_missing(debug=False)
    print("✓ Synonyms-augmented files created")


if __name__ == "__main__":
    main()

