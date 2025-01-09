import numpy as np
import pandas as pd


def cal_signal(zscore_series: pd.Series, upper: float, lower: float) -> pd.Series:
    # 计算前一时刻的zscore值
    zscore_shifted = zscore_series.shift(1)
    signal_series = pd.Series(np.nan, index=zscore_series.index, dtype='string')
    # 上穿1 -> short
    signal_series[(zscore_shifted <= upper) & (zscore_series > upper)] = 'short'
    # 下穿-1 -> long
    signal_series[(zscore_shifted >= lower) & (zscore_series < lower)] = 'long'
    # 上穿0 -> exit_long
    signal_series[(zscore_shifted <= 0) & (zscore_series > 0)] = 'exit_long'
    # 下穿0 -> exit_short
    signal_series[(zscore_shifted >= 0) & (zscore_series < 0)] = 'exit_short'
    return signal_series


def cal_position(signal_series: pd.Series):
    position_series = pd.Series(np.NaN, index=signal_series.index, dtype='int')
    # 计算 position 变化
    position_series[signal_series == 'long'] = 1  # 当 signal 是 long，下一时间点为多仓
    position_series[signal_series == 'short'] = -1  # 当 signal 是 short，下一时间点为空仓
    # 平仓信号：exit_long 在之前是 long，exit_short 在之前是 short
    position_series[signal_series == 'exit_long'] = 0  # exit_long 需要平仓
    position_series[signal_series == 'exit_short'] = 0  # exit_short 需要平仓
    # 将前一时刻的 position 持续更新，直到平仓信号出现
    position_series = position_series.ffill()  # 向前填充 position 序列，直到下一个信号触发

    return position_series.shift()


if __name__ == '__main__':
    # 模拟zscore数据（根据一些波动，产生模拟的信号）
    zscore = pd.Series(
        [0.5, 1.2, -0.8, 0.3, 1.5, -0.2, 0.9, -1.2, 0.1, -0.3, 0.7, -1.0, 1.8, 0.4, -0.4, -1.5, 1.0, -0.6, 0.3, 1.3],
        index=pd.date_range(start='2025-01-01 00:00:00', periods=20, freq='1h')
    )

    signal = cal_signal(zscore, 1, -1)
    position = cal_position(signal)
    df = pd.DataFrame({
        'zscore': zscore,
        'signal': signal,
        'pos': position
    })

    # 打印结果
    print(df)
