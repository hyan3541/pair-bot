from typing import Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


def cal_zscore(spread: pd.Series, window: int = 24) -> pd.Series:
    mean = spread.rolling(window=window).mean()
    std = spread.rolling(window=window).std()
    zscore = (spread - mean) / std
    return zscore


def cal_spread(base: pd.Series, target: pd.Series, hedge_ratio: float) -> pd.Series:
    return target - base * hedge_ratio


def cal_cointegration(base: pd.Series, target: pd.Series) -> Dict:
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
    critical_values = adf_result[4]
    # 判断是否协整 (通常 p-value < 0.05 表示残差平稳，即协整)
    is_coint = p_value < 0.05
    # 返回结果
    return {
        'is_coint': is_coint,
        'adf_statistic': adf_statistic,
        'p_value': round(p_value, 5),
        'hedge_ratio': round(model.params.iloc[1], 5),
        'intercept': round(model.params.iloc[0], 5)
    }


def cal_zero_crossings(spread: pd.Series) -> int:
    zero_crossings = len(np.where(np.diff(np.sign(spread)))[0])
    return zero_crossings


def cal_half_life(spread: pd.Series) -> float:
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


def extract_col(df: pd.DataFrame, col: str, start_time, end_time) -> pd.Series | None:
    # 过滤数据
    filtered_df = df[(df['candle_begin_time'] >= start_time) & (df['candle_begin_time'] <= end_time)]
    # 检查数据是否为空、开头或结尾是否缺失、是否有空值
    if (filtered_df.empty or
            filtered_df['candle_begin_time'].iloc[0] != start_time or
            filtered_df['candle_begin_time'].iloc[-1] != end_time or
            filtered_df[col].isna().any()):
        return None
    return filtered_df[col].reset_index(drop=True)


def process_pair(args):
    symbol1, symbol2, base, target = args
    if base is None or target is None or base.size != target.size:
        print(f'--{symbol1} {symbol2} skip')
        return None
    summary = cal_cointegration(base, target)
    if summary['is_coint']:
        summary['base'] = symbol1
        summary['target'] = symbol2
        spread = cal_spread(base, target, summary['hedge_ratio'])
        summary['zero_crossings'] = cal_zero_crossings(spread)
        summary['half_life'] = cal_half_life(spread)
        print(f'{symbol1} {symbol2} coint')
        return summary
    print(f'--{symbol1} {symbol2} not coint')
    return None






