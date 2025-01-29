from typing import Dict

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from scipy.stats import spearmanr

import statsmodels.api as sm

from program.common import cal_spread


def cal_cointegration(base: pd.Series, target: pd.Series) -> Dict:
    """
    币对协整性判断
    :param base: x
    :param target: y
    :return: summary, 是否协整、adf值、p_value值、对冲比率、线性回归截距
    """
    # 回归分析
    # 添加常数项
    X = sm.add_constant(base)
    # 拟合线性回归模型
    model = sm.OLS(target, X).fit()
    # 获取残差
    residuals = model.resid

    # 残差平稳性检验 (ADF 检验)
    adf_result = adfuller(residuals, autolag='AIC')
    # 提取 ADF 检验结果，adf_statistic
    adf_statistic = adf_result[0]
    p_value = adf_result[1]
    # 判断是否协整 (通常 p-value < 0.05 表示残差平稳，即协整)
    is_coint = p_value < 0.05
    corr, p_value = spearmanr(base, target)
    # 返回结果
    return {
        'is_coint': is_coint,
        'adf_statistic': adf_statistic,
        'p_value': round(p_value, 5),
        'hedge_ratio': round(model.params.iloc[1], 5),
        'intercept': round(model.params.iloc[0], 5),
        'corr': corr
    }


def cal_zero_crossings(spread: pd.Series) -> int:
    """
    价差穿过0轴次数
    :param spread: 价差
    :return: 次数
    """
    zero_crossings = len(np.where(np.diff(np.sign(spread)))[0])
    return zero_crossings


def cal_half_life(spread: pd.Series) -> float:
    """
    半衰期计算，波峰波谷回到0轴的平均时间
    :param spread:
    :return:
    """
    # 计算一阶差分
    delta_spread = spread.diff().dropna()
    # 构造回归模型
    lagged_spread = spread.shift(1).dropna()
    delta_spread = delta_spread.loc[lagged_spread.index]  # 对齐索引
    X = sm.add_constant(lagged_spread)  # 添加常数项
    model = sm.OLS(delta_spread, X).fit()
    # 提取回归系数
    beta = model.params.iloc[1]
    # 计算半衰期
    half_life = -np.log(2) / beta
    return round(half_life, 5)


def process_pair(args):
    """
    币对协整分析
    :param args: symbol1, symbol2, base, target，两个价格序列
    :return: 协整则返回 p_value、zero-crossing、半衰期，否则返回None
    """
    symbol1, symbol2, base, target = args
    if base is None or target is None or base.size != target.size:
        print(f'--{symbol1} {symbol2} skip')
        return None
    corr, p_value = spearmanr(base, target)
    if p_value > 0.05:
        print(f'{symbol1} {symbol2} spearman p_value > 0.05')
        return None
    summary = cal_cointegration(base, target)
    if summary['is_coint']:
        summary['base'] = symbol1
        summary['target'] = symbol2
        spread = cal_spread(base, target, summary['hedge_ratio'])
        summary['zero_crossings'] = cal_zero_crossings(spread)
        summary['half_life'] = cal_half_life(spread)
        summary['spearman'] = round(corr, 3)
        print(f'{symbol1} {symbol2} coint')
        return summary
    print(f'--{symbol1} {symbol2} not coint')
    return None
