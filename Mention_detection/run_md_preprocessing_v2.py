# import os
# from transformers import BertTokenizerFast
# from data_processing.mention_detection.md_preprocessing_v2 import MentionDetectionPreprocessingV2

# def process_all_splits():
#     """Process all combinations of debug/full and train/val/test splits"""
    
#     tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
#     # Process both debug and full data
#     for debug_mode in [True, False]:
#         debug_str = "debug" if debug_mode else "full"
#         print(f"\n{'='*60}")
#         print(f"Processing {debug_str.upper()} data")
#         print('='*60)
        
#         # Process train, val, and test splits separately
#         for split in ['train', 'val', 'test']:
#             print(f"\n--- Processing {split} split for {debug_str} data ---")
            
#             md = MentionDetectionPreprocessingV2(debug=debug_mode, split_type=split)
            
#             # Create the train/val/test splits if they don't exist
#             md.create_train_val_test_splits()
            
#             # Parse MedMention data to JSON
#             if not (os.path.exists(md.abstract_dict_filepath) and 
#                     os.path.exists(md.spans_dict_filepath)):
#                 md.parse_medmention_to_json()
#             else:
#                 print(f"JSON files already exist for {split} {debug_str}")
            
#             # Build ID to CUI/mention mapping
#             if not os.path.exists(md.id2cuistr_filepath):
#                 md.build_id2cuistr()
#             else:
#                 print(f"id2cuistr already exists for {split} {debug_str}")
            
#             # Create tokenized data with BIO labels
#             if not os.path.exists(md.tokenized_filepath):
#                 md.bio_labelling(tokenizer)
#             else:
#                 print(f"Tokenized data already exists for {split} {debug_str}")
    
#     print("\n" + "="*60)
#     print("All preprocessing complete!")
#     print("="*60)
    
#     # Print summary of created files
#     print("\nCreated files summary:")
#     for debug_mode in [True, False]:
#         debug_str = "debug" if debug_mode else "full"
#         print(f"\n{debug_str.upper()} data:")
#         for split in ['train', 'val', 'test']:
#             split_dir = f"data/processed/mention_detection/{split}"
#             if os.path.exists(split_dir):
#                 files = [f for f in os.listdir(split_dir) if debug_str in f]
#                 print(f"  {split}: {len(files)} files")

# if __name__ == "__main__":
#     process_all_splits()

# # Run with: python -m data_processing.mention_detection.run_md_preprocessing_v2


import os
import argparse
from transformers import BertTokenizerFast
from data_processing.mention_detection.md_preprocessing_v2 import MentionDetectionPreprocessingV2

def process_specific_split(debug_mode, split_type, tokenizer):
    """Process a specific combination of debug/full and train/val/test split"""
    
    debug_str = "debug" if debug_mode else "full"
    print(f"\n{'='*60}")
    print(f"Processing {debug_str.upper()} data - {split_type.upper()} split")
    print('='*60)
    
    # Create the train/val/test splits if they don't exist.
    md = MentionDetectionPreprocessingV2(debug=debug_mode, split_type=split_type)
    md.create_train_val_test_splits()
    
    # Parse MedMention data to JSON.
    if not (os.path.exists(md.abstract_dict_filepath) and 
            os.path.exists(md.spans_dict_filepath)):
        md.parse_medmention_to_json()
    else:
        print(f"JSON files already exist for {split_type} {debug_str}")
    
    # Build ID to CUI/mention mapping
    if not os.path.exists(md.id2cuistr_filepath):
        md.build_id2cuistr()
    else:
        print(f"id2cuistr already exists for {split_type} {debug_str}")
    
    # Create tokenized data with BIO labels
    if not os.path.exists(md.tokenized_filepath):
        md.bio_labelling(tokenizer)
    else:
        print(f"Tokenized data already exists for {split_type} {debug_str}")
    
    print(f"\nCompleted processing {debug_str} {split_type} split!")

def process_all_splits():
    """Process all combinations of debug/full and train/val/test splits"""
    
    tokenizer = BertTokenizerFast.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    
    # Process both debug and full data
    for debug_mode in [True, False]:
        debug_str = "debug" if debug_mode else "full"
        print(f"\n{'='*60}")
        print(f"Processing {debug_str.upper()} data")
        print('='*60)
        
        # Process train, val, and test splits separately
        for split in ['train', 'val', 'test']:
            print(f"\n--- Processing {split} split for {debug_str} data ---")
            
            md = MentionDetectionPreprocessingV2(debug=debug_mode, split_type=split)
            
            # Create the train/val/test s plits if they don't exist
            md.create_train_val_test_splits()
            
            # Parse MedMention data to JSON
            if not (os.path.exists(md.abstract_dict_filepath) and 
                    os.path.exists(md.spans_dict_filepath)):
                md.parse_medmention_to_json()
            else:
                print(f"JSON files already exist for {split} {debug_str}")
            
            # Build ID to CUI/mention mapping
            if not os.path.exists(md.id2cuistr_filepath):
                md.build_id2cuistr()
            else:
                print(f"id2cuistr already exists for {split} {debug_str}")
            
            # Create tokenized data with BIO labels
            if not os.path.exists(md.tokenized_filepath):
                md.bio_labelling(tokenizer)
            else:
                print(f"Tokenized data already exists for {split} {debug_str}")
    
    print("\n" + "="*60)
    print("All preprocessing complete!")
    print("="*60)
    
    # Print summary of created files
    print("\nCreated files summary:")
    for debug_mode in [True, False]:
        debug_str = "debug" if debug_mode else "full"
        print(f"\n{debug_str.upper()} data:")
        for split in ['train', 'val', 'test']:
            split_dir = f"data/processed/mention_detection/{split}"
            if os.path.exists(split_dir):
                files = [f for f in os.listdir(split_dir) if debug_str in f]
                print(f"  {split}: {len(files)} files")

def main():
    parser = argparse.ArgumentParser(description='Process MedMention data for mention detection')
    parser.add_argument('--mode', choices=['debug', 'full'], 
                       help='Choose debug or full data mode')
    parser.add_argument('--split', choices=['train', 'val', 'test'], 
                       help='Choose specific split to process')
    parser.add_argument('--all', action='store_true', 
                       help='Process all combinations (original behavior)')
    
    args = parser.parse_args()
    
    tokenizer = BertTokenizerFast.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

    if args.all:
        # Original behavior - process all combinations
        process_all_splits()
    elif args.mode and args.split:
        # Process specific combination
        debug_mode = args.mode == 'debug'
        process_specific_split(debug_mode, args.split, tokenizer)
    else:
        print("Usage examples:")
        print("  # Process all combinations:")
        print("  python -m data_processing.mention_detection.run_md_preprocessing_v2 --all")
        print()
        print("  # Process specific combination:")
        print("  python -m data_processing.mention_detection.run_md_preprocessing_v2 --mode debug --split train")
        print("  python -m data_processing.mention_detection.run_md_preprocessing_v2 --mode full --split val")
        print("  python -m data_processing.mention_detection.run_md_preprocessing_v2 --mode debug --split test")
        print()
        print("Available options:")
        print("  --mode: debug or full")
        print("  --split: train, val, or test")
        print("  --all: process all 6 combinations")

if __name__ == "__main__":
    main()

# Run with: python -m data_processing.mention_detection.run_md_preprocessing_v2 --debug