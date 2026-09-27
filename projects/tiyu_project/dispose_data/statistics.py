#!/usr/bin/env python3
import csv
import sys
from collections import Counter

def main():
    if len(sys.argv) != 2:
        print("用法: python count_tf.py <csv文件路径>")
        sys.exit(1)

    csv_file = sys.argv[1]
    counter = Counter()

    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # 检查是否包含 qc_ai 列
            if 'qc_ai' not in reader.fieldnames:
                print("错误：CSV 文件中缺少 'qc_ai' 列")
                sys.exit(1)
            
            for row in reader:
                value = row['qc_ai'].strip()
                if value:                     # 忽略空值
                    counter[value] += 1
                    
    except FileNotFoundError:
        print(f"错误：文件 '{csv_file}' 未找到")
        sys.exit(1)
    except Exception as e:
        print(f"读取文件时出错: {e}")
        sys.exit(1)

    # 输出统计结果
    print("统计结果：")
    for key, count in sorted(counter.items()):
        print(f"  {key}: {count}")
    total = sum(counter.values())
    print(f"总计: {total}")

if __name__ == "__main__":
    main()
