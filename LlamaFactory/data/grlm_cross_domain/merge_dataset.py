import json
import os

def merge_specific_json_files(file_list, output_file):
    """
    合并指定的JSON文件列表
    
    Args:
        file_list: 要合并的文件路径列表
        output_file: 输出文件名
    """
    
    all_data = []
    
    for file_path in file_list:
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                if isinstance(data, list):
                    all_data.extend(data)
                    print(f"成功读取 {file_path}: {len(data)} 条数据")
                else:
                    print(f"文件 {file_path} 的数据格式不是列表，跳过")
                    
        except Exception as e:
            print(f"读取 {file_path} 时出错: {e}")
    
    # 保存合并后的数据
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n合并完成！")
        print(f"输入文件数量: {len(file_list)}")
        print(f"总数据条数: {len(all_data)}")
        print(f"输出文件: {output_file}")
        
    except Exception as e:
        print(f"保存文件时出错: {e}")

# 使用方法
if __name__ == "__main__":
    files_to_merge = [
        "amazon_electronic_phone_sft_data_rec_simplified.json", 
        "amazon_electronic_phone_sft_data_meta2tid.json"
    ]
    merge_specific_json_files(files_to_merge, "amazon_electronic_phone_sft_data_rec_meta2sid.json")

    files_to_merge = [
        "amazon_cloth_sport_sft_data_rec_simplified.json", 
        "amazon_cloth_sport_sft_data_meta2tid.json"
    ]
    merge_specific_json_files(files_to_merge, "amazon_cloth_sport_sft_data_rec_meta2sid.json")