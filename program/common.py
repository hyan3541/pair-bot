import pandas as pd


def cal_spread(base: pd.Series, target: pd.Series, hedge_ratio: float) -> pd.Series:
    """
    价差计算
    :param base: 价格序列 x
    :param target: 价格序列 y
    :param hedge_ratio: 对冲比率
    :return: 价差序列 = y - h * x
    """
    return target - base * hedge_ratio


def extract_col(df: pd.DataFrame, col: str, start_time, end_time) -> pd.Series | None:
    """
    从 df 提取一段 pd.Series，pd.Series计算比带着df计算快很多
    :param df: dataframe
    :param col: 列名
    :param start_time: 序列开始时间
    :param end_time: 序列结束时间
    :return: 提取后的序列，完整覆盖时间区间，没有空值
    """
    # 过滤数据
    filtered_df = df[(df['candle_begin_time'] >= start_time) & (df['candle_begin_time'] <= end_time)]
    # 检查数据是否为空、开头或结尾是否缺失、是否有空值
    if (filtered_df.empty or
            filtered_df['candle_begin_time'].iloc[0] != start_time or
            filtered_df['candle_begin_time'].iloc[-1] != end_time or
            filtered_df[col].isna().any()):
        return None
    return filtered_df[col].reset_index(drop=True)
