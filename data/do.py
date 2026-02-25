import json

def process_health_data(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 1. 删除 label 为 2 的数据
        filtered_data = [item for item in data if item.get("label") != 2]
        
        for index, item in enumerate(filtered_data):
            # 2. 重新编号 id (从 1 开始)
            item["id"] = str(index + 1)
            
            # 3. 对 evidence 内容进行补齐，确保有 5 条记录
            evidence = item.get("evidence", {})
            current_len = len(evidence)
            
            if current_len < 5:
                for i in range(current_len, 5):
                    # 以字符串索引作为 key，保持数据结构一致性
                    evidence[str(i)] = {"content": ""}
            
            item["evidence"] = evidence

        # 写入新文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=4)
            
        print(f"处理完成。原始数据量: {len(data)}，处理后数据量: {len(filtered_data)}")

    except Exception as e:
        print(f"处理过程中出错: {e}")

# 执行处理
if __name__ == "__main__":
    process_health_data('data/health_info_retrieved.json', 'data/health_info_retrieved.json')