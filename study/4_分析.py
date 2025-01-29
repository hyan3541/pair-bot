import joblib
import pandas as pd

def len_dist(s: pd.Series):
    group_marker = s.ne(s.shift()).cumsum()

    # 2. 过滤出 True 的组，并计算每组长度
    true_groups = s.groupby(group_marker).sum()

    # 3. 统计不同长度的分布
    length_distribution = true_groups.value_counts().sort_index()
    print(length_distribution)

if __name__ == '__main__':
    data = joblib.load('coint720.pkl')
    stat_list = []
    for info in data:
        stat = {'pair': info['pair'], 'sum': info['coint'].astype(int).sum()}
        stat_list.append(stat)
        len_dist(info['coint'])
        break
    exit()
    df = pd.DataFrame(stat_list)
    df = df.sort_values(by='sum', ascending=False)
    df.to_csv('rank168.csv')
