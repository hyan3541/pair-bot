import time

import joblib
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from statsmodels.tsa.stattools import adfuller


def cal_coint(base: pd.Series, target: pd.Series):
    """
    币对协整性判断
    :param base: x
    :param target: y
    :return: true/false
    """
    # 回归分析
    # 添加常数项
    X = sm.add_constant(base)
    # 拟合线性回归模型
    model = sm.OLS(target, X).fit()
    # 获取残差
    residuals = model.resid
    # adf检测
    adf_result = adfuller(residuals, autolag='AIC')
    p_value = adf_result[1]
    return p_value < 0.05 and adf_result[0] < adf_result[4]['5%']


def process_pair(a, b, s1: pd.Series, s2: pd.Series, window: int = 24 * 30):
    """
    滚动计算协整关系
    :param s1: 价格序列 1
    :param s2: 价格序列 2
    :param window: 滚动窗口大小
    :return: 协整结果序列 (布尔值)
    """
    # 初始化结果列表
    coint_results = []
    begin = time.time()
    # 滚动窗口计算
    for i in range(window, len(s1)):
        base = s1.iloc[i - window:i]  # 取窗口内的 base 数据
        target = s2.iloc[i - window:i]  # 取窗口内的 target 数据
        result = cal_coint(base, target)  # 计算协整关系
        coint_results.append(result)
    print(f'{a}_{b} cost {time.time() - begin}')
    # 将结果转换为 Series，索引与原始数据对齐
    return {'pair': f'{a}_{b}', 'coint': pd.Series(coint_results, index=s1.index[window:])}


if __name__ == '__main__':
    data = joblib.load('data2024.pkl')
    pairs = joblib.load('pairs.pkl')
    arg_list = []
    for a, b in pairs:
        close1 = data[a]['close'].reset_index(drop=True)
        close2 = data[b]['close'].reset_index(drop=True)
        arg_list.append((a, b, close1, close2))
    begin = time.time()

    results = Parallel(n_jobs=16)(delayed(process_pair)(a, b, close1, close2) for a, b, close1, close2 in arg_list)
    joblib.dump(results, 'coint720.pkl')
    print(f'cost {time.time() - begin}')
