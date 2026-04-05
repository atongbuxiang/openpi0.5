# env: pandas, pyarrow
import pandas as pd
import os, shutil, glob, json

IN_DIR  = "/root/data0/RynnVLA-001/datasets/so100_pickplace/so100_pickplace_1/data/chunk-000" 
OUT_DIR = "/root/data0/RynnVLA-001/datasets/temp/so100_pickplace"            

# 可选：干净一点
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# 获取所有parquet文件
parquet_files = glob.glob(os.path.join(IN_DIR, "*.parquet"))
print(f"发现 {len(parquet_files)} 个parquet文件: {[os.path.basename(f) for f in parquet_files]}")

# 逐个处理每个parquet文件
for parquet_file in parquet_files:
    # 获取文件名（不包含扩展名）
    base_name = os.path.splitext(os.path.basename(parquet_file))[0]
    
    print(f"正在处理: {base_name}.parquet")
    
    # 使用pandas加载parquet文件
    df = pd.read_parquet(parquet_file)
    
    # 输出jsonl文件，前缀名相同
    output_file = os.path.join(OUT_DIR, f"{base_name}.jsonl")
    
    # 重置timestamp为从0.0开始，按10Hz（0.1秒间隔）累加
    if 'timestamp' in df.columns:
        num_examples = len(df)
        # 从0.0开始，每个间隔0.1秒（10Hz）
        df['timestamp'] = [i * 0.1 for i in range(num_examples)]
    
    # 导出为jsonl
    df.to_json(output_file, orient="records", lines=True)
    
    print(f"已生成: {base_name}.jsonl")

print(f"所有转换完成! 输出目录: {OUT_DIR}")
