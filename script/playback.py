from scipy.stats import spearmanr

import config
from program.analyse import cal_cointegration
from program.common import cal_spread, extract_cols
from program.evaluate import cal_evaluate, plot_output
from program.function import *
from program.curve import *
from program.rolling import *

pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.expand_frame_repr', False)  # 不换行


if __name__ == '__main__':
    data = pd.read_pickle(config.swap_path)
    symbol1 = 'CELO-USDT'
    symbol2 = 'BAT-USDT'
    start_date = '2024-01-01'
    end_date = '2024-12-01'
    cols = ['candle_begin_time', 'close', 'high', 'open', 'low']
    # 获得2个价格序列
    base = extract_cols(data[symbol1], cols, pd.Timestamp(start_date), pd.Timestamp(end_date))
    target = extract_cols(data[symbol2], cols, pd.Timestamp(start_date), pd.Timestamp(end_date))
    base_close = base['close']
    target_close = target['close']
    # 滚动协整、滚动相关系数
    coint_list, h_ratio_list, corr_list = rolling_coint(base_close, target_close, 720)

    spread = cal_spread(base_close, target_close, hedge_ratio=h_ratio_list)
    zscore_series = cal_zscore(spread, 168)
    signal = cal_signal(zscore_series, coint_list,2, -2)

    df = pd.concat([base_close, target_close, coint_list, h_ratio_list, corr_list, spread, zscore_series, signal], axis=1)
    print(df)
    # exit()
    target_pos = cal_position(signal)
    base_pos = invert_position(target_pos)
    target['pos'] = target_pos
    base['pos'] = base_pos
    df1 = cal_equity_curve(base, leverage_rate=2)
    df2 = cal_equity_curve(target, leverage_rate=2)
    equity_curve = merge_curve(df1['equity_curve'], df2['equity_curve'], base['candle_begin_time'],
                               1, 1)
    data = cal_evaluate(equity_curve)
    print(data)
    plot_output(equity_curve, data, config.plot_path)
