from typing import Dict
import pandas as pd
from joblib import Parallel, delayed

import config
import time
import itertools

from program.analyse import process_pair
from program.common import extract_col


def get_cointegrated_pairs(all_df: Dict[str, pd.DataFrame],
                           start_time: pd.Timestamp, end_time: pd.Timestamp, n_jobs=-1, parallel=True):
    symbols = all_df.keys()
    # 进行币对组合，排除自身组合
    combinations = list(itertools.combinations(symbols, 2))

    # 构建入参列表
    arg_list = []
    all_df_range = {}
    for symbol, df in all_df.items():
        all_df_range[symbol] = extract_col(all_df[symbol], 'close', start_time, end_time)

    for symbol1, symbol2 in combinations:
        base = all_df_range[symbol1]
        target = all_df_range[symbol2]
        arg_list.append((symbol1, symbol2, base, target))

    # 批量协整行分析
    if parallel:
        # 使用 joblib 并行处理
        results = Parallel(n_jobs=n_jobs)(delayed(process_pair)(arg) for arg in arg_list)
    else:
        # 串行处理
        results = [process_pair(arg) for arg in arg_list]
    # 过滤None值
    coint_pair_list = [res for res in results if res is not None]
    if coint_pair_list:
        df_coint = pd.DataFrame(coint_pair_list)
        # 按 zero_crossings 排序
        df_coint = df_coint.sort_values("zero_crossings", ascending=False)
        # 保存到 csv 文件
        df_coint.to_csv("coint_pairs.csv", index=False)
    else:
        df_coint = pd.DataFrame()
        print("No cointegrated pairs found.")

    return df_coint


if __name__ == '__main__':
    data = pd.read_pickle(config.swap_path)
    begin = time.time()
    start_date = '2024-10-01'
    end_date = '2024-11-01'
    get_cointegrated_pairs(all_df=data, start_time=pd.Timestamp(start_date), end_time=pd.Timestamp(end_date))
    print(f'cost {time.time() - begin}')
