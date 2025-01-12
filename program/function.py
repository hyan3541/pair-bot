import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.expand_frame_repr', False)  # 不换行


def cal_zscore(spread: pd.Series, window: int = 24) -> pd.Series:
    """
    z-score计算
    :param spread: 价差序列
    :param window: 窗口
    :return: 基于价差序列开窗计算的结果
    """
    mean = spread.rolling(window=window).mean()
    std = spread.rolling(window=window).std()
    zscore = (spread - mean) / std
    return zscore


def cal_signal(zscore_series: pd.Series, upper: float, lower: float) -> pd.DataFrame:
    """
    交易信号计算
    zscore 上穿0，平多
    zscore 下穿0，平空
    zscore 上穿上界，开空
    zscore 下穿下界，开多
    会返回4列，没有合为一列，因为平多和开空、或者平空和开多可能同时发生，两个信号共存，仓位反转
    :param zscore_series: zscore序列
    :param upper: 上界
    :param lower: 下界
    :return: df，包含开多、平多、开空、平空的时机
    """
    # 计算前一时刻的zscore值
    zscore_shifted = zscore_series.shift(1)
    # 上穿0 -> exit_long
    exit_long = (zscore_shifted <= 0) & (zscore_series > 0)
    # 下穿0 -> exit_short
    exit_short = (zscore_shifted >= 0) & (zscore_series < 0)
    # 上穿1 -> short
    short = (zscore_shifted <= upper) & (zscore_series > upper)
    # 下穿-1 -> long
    long = (zscore_shifted >= lower) & (zscore_series < lower)

    return pd.DataFrame({
        'long': long,
        'short': short,
        'exit_long': exit_long,
        'exit_short': exit_short
    })


def cal_position(signals: pd.DataFrame):
    """
    根据信号计算仓位，考虑平仓和开仓信号同时发生的情况
    :param signals: 包含开多、平多、开空、平空信号的DataFrame
    :return: 仓位序列，1表示做多，-1表示做空，0表示空仓
    """
    # 初始化仓位序列，初始状态为0（空仓）
    position = pd.Series(0, index=signals.index)

    # 遍历信号
    for i in range(1, len(signals)):
        # 当前仓位状态
        current_position = position.iloc[i - 1]

        # 信号
        long = signals['long'].iloc[i]
        short = signals['short'].iloc[i]
        exit_long = signals['exit_long'].iloc[i]
        exit_short = signals['exit_short'].iloc[i]

        # 处理信号
        if current_position == 1:  # 当前是做多 long 和 exit_short 无效
            if exit_long:  # 平多
                current_position = 0
            if short:  # 做空信号（平多后开空）
                current_position = -1
        elif current_position == -1:  # 当前是做空 short 和 exit_long 无效
            if exit_short:  # 平空
                current_position = 0
            if long:  # 做多信号（平空后开多）
                current_position = 1
        else:  # 当前是空仓，exit_short 和 exit_long 无效
            if long:  # 做多信号
                current_position = 1
            elif short:  # 做空信号
                current_position = -1

        # 更新仓位
        position.iloc[i] = current_position
    # 仓位在出现信号后下根K线调整
    position = position.shift()
    # 最后一根k线平仓
    position.iloc[-1] = 0
    return position


def invert_position(position_series: pd.Series) -> pd.Series:
    """
    配对交易，对手合约的相反仓位
    :param position_series: target的仓位
    :return: base的仓位
    """
    invert_series = np.where(position_series == 1, -1, np.where(position_series == -1, 1, 0))
    return pd.Series(invert_series)
