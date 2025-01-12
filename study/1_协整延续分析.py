import pandas as pd

# 读取不同月份、年份（eg 11月和12月）或者不同周期（eg 1个月和1年）的协整对
df_a = pd.read_csv('coint_pairs2022.csv')
df_b = pd.read_csv('coint_pairs2023.csv')

# 重命名 p_value 列
df_a = df_a.rename(columns={'p_value': 'p_value_a'})
df_b = df_b.rename(columns={'p_value': 'p_value_b'})

# 合并两个 DataFrame，基于 base 和 target 列
merged_df = pd.merge(df_a, df_b, on=['base', 'target'], how='inner')

# 筛选需要的列
result_df = merged_df[['base', 'target', 'p_value_a', 'p_value_b']].sort_values(by=['base', 'target'])

# 不同月都协整，或者月维度、年维度都协整的币对，保存到 c.csv
result_df.to_csv('2223.csv', index=False)

print("结果已保存到 c.csv")
