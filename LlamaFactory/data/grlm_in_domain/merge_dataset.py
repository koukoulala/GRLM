import json
import os

def merge_specific_json_files(file_list, output_file):
    
    all_data = []
    
    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"not found: {file_path}")
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if isinstance(data, list):
                    all_data.extend(data)
                    print(f"read {file_path}: {len(data)} data")
                else:
                    print(f"file {file_path} is not list")
                    
        except Exception as e:
            print(f"Wrong with {file_path} : {e}")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nFinshed")
        print(f"input data: {len(file_list)}")
        print(f"total data: {len(all_data)}")
        print(f"output file: {output_file}")
        
    except Exception as e:
        print(f"Wrong with: {e}")

if __name__ == "__main__":
    files_to_merge = [
        "amazon_beauty_sft_data_rec_simplified.json", 
        "amazon_beauty_sft_data_meta2tid.json",
        "amazon_beauty_sasrec_collaborative_sft.json"
    ]
    merge_specific_json_files(files_to_merge, "amazon_beauty_sft_data_combined.json")
    files_to_merge = [
        "amazon_sports_sft_data_rec_simplified.json", 
        "amazon_sports_sft_data_meta2tid.json",
        "amazon_sports_sasrec_collaborative_sft.json"
    ]
    merge_specific_json_files(files_to_merge, "amazon_sports_sft_data_combined.json")
    files_to_merge = [
        "amazon_toys_sft_data_rec_simplified.json", 
        "amazon_toys_sft_data_meta2tid.json",
        "amazon_toys_sasrec_collaborative_sft.json"
    ]
    merge_specific_json_files(files_to_merge, "amazon_toys_sft_data_combined.json")