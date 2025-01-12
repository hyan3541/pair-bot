from typing import Dict

import numpy as np
import pandas as pd

import config


def cal_min_qty() -> Dict:
    min_qty_df = pd.read_csv(config.min_qty_path, encoding='gbk')
    min_qty_df['合约'] = min_qty_df['合约'].str.replace('-', '')
    min_qty_df['最小下单量'] = -np.log10(min_qty_df['最小下单量']).round().astype(int)
    min_qty_df.set_index('合约', inplace=True)
    return min_qty_df['最小下单量'].to_dict()


def cal_equity_curve(df: pd.DataFrame, slippage: float = 1 / 1000, c_rate: float = 5 / 10000,
                     leverage_rate: float = 1, min_amount: float = 0.01,
                     min_margin_ratio: float = 1 / 100, initial_cash: float = 1000) -> pd.DataFrame:
    """
    邢大资金曲线计算方法
    :param df: 含 candle_begin_time open close high low pos
    :param slippage:  滑点 ，可以用百分比，也可以用固定值。建议币圈用百分比，股票用固定值
    :param c_rate:  手续费，commission fees，默认为万分之5。不同市场手续费的收取方法不同，对结果有影响。比如和股票就不一样。
    :param leverage_rate:  杠杆倍数
    :param min_amount:  最小下单量
    :param min_margin_ratio: 最低保证金率，低于就会爆仓
    :param initial_cash:
    :return: curve
    """
    # 下根k线开盘价
    df['next_open'] = df['open'].shift(-1)  # 下根K线的开盘价
    df['next_open'] = df['next_open'].fillna(value=df['close'])

    # 找出开仓、平仓的k线
    condition1 = df['pos'] != 0  # 当前周期不为空仓
    condition2 = df['pos'] != df['pos'].shift(1)  # 当前周期和上个周期持仓方向不一样。
    open_pos_condition = condition1 & condition2

    condition1 = df['pos'] != 0  # 当前周期不为空仓
    condition2 = df['pos'] != df['pos'].shift(-1)  # 当前周期和下个周期持仓方向不一样。
    close_pos_condition = condition1 & condition2

    # 对每次交易进行分组
    df.loc[open_pos_condition, 'start_time'] = df['candle_begin_time']
    df['start_time'] = df['start_time'].ffill()
    df.loc[df['pos'] == 0, 'start_time'] = pd.NaT

    # 开始计算资金曲线
    # 在open_pos_condition的K线，以开盘价计算买入合约的数量。（当资金量大的时候，可以用5分钟均价）
    df.loc[open_pos_condition, 'contract_num'] = initial_cash * leverage_rate / (min_amount * df['open'])
    df['contract_num'] = np.floor(df['contract_num'])  # 对合约张数向下取整
    # 开仓价格：理论开盘价加上相应滑点
    df.loc[open_pos_condition, 'open_pos_price'] = df['open'] * (1 + slippage * df['pos'])
    # 开仓之后剩余的钱，扣除手续费
    df['cash'] = initial_cash - df['open_pos_price'] * min_amount * df['contract_num'] * c_rate  # 即保证金

    # 开仓之后每根K线结束时
    # 买入之后cash，contract_num，open_pos_price不再发生变动
    df['contract_num'] = df['contract_num'].ffill()
    df['open_pos_price'] = df['open_pos_price'].ffill()
    df['cash'] = df['cash'].ffill()
    df.loc[df['pos'] == 0, ['contract_num', 'open_pos_price', 'cash']] = None

    # 在平仓时
    # 平仓价格
    df.loc[close_pos_condition, 'close_pos_price'] = df['next_open'] * (1 - slippage * df['pos'])
    # 平仓之后剩余的钱，扣除手续费
    df.loc[close_pos_condition, 'close_pos_fee'] = df['close_pos_price'] * min_amount * df['contract_num'] * c_rate

    # 计算利润
    # 开仓至今持仓盈亏
    df['profit'] = min_amount * df['contract_num'] * (df['close'] - df['open_pos_price']) * df['pos']
    # 平仓时理论额外处理
    df.loc[close_pos_condition, 'profit'] = min_amount * df['contract_num'] * (
            df['close_pos_price'] - df['open_pos_price']) * df['pos']
    # 账户净值
    df['net_value'] = df['cash'] + df['profit']

    # 计算爆仓
    # 至今持仓盈亏最小值
    df.loc[df['pos'] == 1, 'price_min'] = df['low']
    df.loc[df['pos'] == -1, 'price_min'] = df['high']
    df['profit_min'] = min_amount * df['contract_num'] * (df['price_min'] - df['open_pos_price']) * df['pos']
    # 账户净值最小值
    df['net_value_min'] = df['cash'] + df['profit_min']
    # 计算保证金率
    df['margin_ratio'] = df['net_value_min'] / (min_amount * df['contract_num'] * df['price_min'])
    # 计算爆仓
    df.loc[df['margin_ratio'] <= (min_margin_ratio + c_rate), 'liquidate'] = 1

    # 平仓时扣除手续费
    df.loc[close_pos_condition, 'net_value'] -= df['close_pos_fee']
    # 应对偶然情况：下一根K线开盘价格价格突变，在平仓的时候爆仓。此处处理有省略，不够精确。
    df.loc[close_pos_condition & (df['net_value'] < 0), 'liquidate'] = 1

    # 对爆仓进行处理
    df['liquidate'] = df.groupby('start_time')['liquidate'].ffill()
    df.loc[df['liquidate'] == 1, 'net_value'] = 0

    # 计算资金曲线
    df['equity_change'] = df['net_value'].pct_change(fill_method=None)
    df.loc[open_pos_condition, 'equity_change'] = df.loc[open_pos_condition, 'net_value'] / initial_cash - 1  # 开仓日的收益率
    df['equity_change'] = df['equity_change'].fillna(value=0)
    df['equity_curve'] = (1 + df['equity_change']).cumprod()
    # 删除不必要的数据，并存储
    df.drop(['next_open', 'contract_num', 'open_pos_price', 'cash', 'close_pos_price', 'close_pos_fee',
             'profit', 'net_value', 'price_min', 'profit_min', 'net_value_min', 'margin_ratio', 'liquidate'],
            axis=1, inplace=True)

    return df


def merge_curve(curve1: pd.Series, curve2: pd.Series, cbts, weight1: float, weight2: float) -> pd.DataFrame:
    total_weight = weight1 + weight2
    coef1 = weight1 / total_weight
    coef2 = weight2 / total_weight
    curve = coef1 * curve1 + coef2 * curve2
    return pd.DataFrame({
        'candle_begin_time': cbts,
        'equity_curve': curve
    })
