import pandas as pd

# 读取 a.csv 和 b.csv
df_a = pd.read_csv('a.csv')
df_b = pd.read_csv('b.csv')

# 重命名 p_value 列
df_a = df_a.rename(columns={'p_value': 'p_value_a'})
df_b = df_b.rename(columns={'p_value': 'p_value_b'})

# 合并两个 DataFrame，基于 base 和 target 列
merged_df = pd.merge(df_a, df_b, on=['base', 'target'], how='inner')

# 筛选需要的列
result_df = merged_df[['base', 'target', 'p_value_a', 'p_value_b']]

# 保存到 c.csv
result_df.to_csv('c.csv', index=False)

print("结果已保存到 c.csv")