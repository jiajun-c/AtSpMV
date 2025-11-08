#!/bin/bash
# 文件名：run_all_tests.sh

DATA_DIR="/staff/zhaoyang/database/sparse-matrix"
OUTPUT_DIR="./results_spmv"
SIZE_THRESHOLD_MB=1024

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 遍历data目录下的所有文件夹
for folder in $DATA_DIR/*/; do
    # 获取文件夹名（去掉路径和斜杠）
    folder_name=$(basename "$folder")
    sleep 5
    # 构建.mtx文件路径（假设与文件夹同名）
    mtx_file="$DATA_DIR/$folder_name/$folder_name.mtx"
    # 检查.mtx文件是否存在
    if [ -f "$mtx_file" ]; then
        file_size=$(stat -f%z "$mtx_file" 2>/dev/null || stat -c%s "$mtx_file" 2>/dev/null)
        file_size_mb=$((file_size / 1024 / 1024))
        if [ "$file_size_mb" -gt "$SIZE_THRESHOLD_MB" ]; then
            echo "[$(date '+%H:%M:%S')] ⚠️  跳过: $folder_name (大小: ${file_size_mb}MB > ${SIZE_THRESHOLD_MB}MB)"
            continue
        fi
        echo "正在处理: $folder_name"
        timeout 120 ./build/spmv_double "$mtx_file" 10 >> "speedup_dy.csv"
        echo "已完成: $folder_name"
    else
        echo "警告: $mtx_file 不存在，跳过"
    fi
done

echo "所有测试完成！"