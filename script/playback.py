import config
from program.analyse import cal_cointegration
from program.common import cal_spread, extract_cols
from program.evaluate import cal_evaluate, plot_output
from program.function import *
from program.curve import *

if __name__ == '__main__':
    data = pd.read_pickle(config.swap_path)
    symbol1 = 'ENJ-USDT'
    symbol2 = 'HIGH-USDT'
    start_date = '2024-10-01'
    end_date = '2024-11-01'
    cols = ['candle_begin_time', 'close', 'high', 'open', 'low']
    base = extract_cols(data[symbol1], cols, pd.Timestamp(start_date), pd.Timestamp(end_date))
    target = extract_cols(data[symbol2], cols, pd.Timestamp(start_date), pd.Timestamp(end_date))
    base_close = base['close']
    target_close = target['close']
    summary = cal_cointegration(base_close, target_close)
    print(summary)
    spread = cal_spread(base_close, target_close, hedge_ratio=summary['hedge_ratio'])
    zscore_series = cal_zscore(spread)
    signal = cal_signal(zscore_series, 2, -2)
    target_pos = cal_position(signal)
    base_pos = invert_position(target_pos)
    target['pos'] = target_pos
    base['pos'] = base_pos
    df1 = cal_equity_curve(base, leverage_rate=3)
    df2 = cal_equity_curve(target, leverage_rate=3)
    equity_curve = merge_curve(df1['equity_curve'], df2['equity_curve'], base['candle_begin_time'],
                               1, 1)
    data = cal_evaluate(equity_curve)
    print(data)
    plot_output(equity_curve, data, config.plot_path)
