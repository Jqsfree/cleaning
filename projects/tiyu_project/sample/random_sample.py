import pandas as pd

# 读取数据
df = pd.read_csv("/home/jqs/tiyu/dispose_data/final_baseball_videos.csv")

# =====================
# 随机抽样
# =====================
"""
 抽样数量
SAMPLE_SIZE = 

# 随机抽样
sample_df = df.sample(
    n=min(SAMPLE_SIZE, len(df)),
    random_state=42
)
"""
sample_df = df.sample(
    frac=0.1,       # 10%
    random_state=42
)

# 保存抽样结果
sample_df.to_csv("/home/jqs/tiyu/dispose_data/sample_videos.csv",
    index=False
)

print("抽样完成")
print("抽样数量:", len(sample_df))
