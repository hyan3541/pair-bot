import config
from functions import *
from func import *
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)  # 显示所有行
pd.set_option('display.expand_frame_repr', False)  # 不换行
if __name__ == '__main__':
    data = pd.read_pickle(config.swap_path)
    symbol1 = 'OP-USDT'
    symbol2 = 'IOTX-USDT'
    start_date = '2024-11-01'
    end_date = '2024-12-01'
    base = extract_col(data[symbol1], 'close', pd.Timestamp(start_date), pd.Timestamp(end_date))
    target = extract_col(data[symbol2], 'close', pd.Timestamp(start_date), pd.Timestamp(end_date))

    summary = cal_cointegration(base, target)
    print(summary)
    spread = cal_spread(base, target, hedge_ratio=summary['hedge_ratio'])
    zscore_series = cal_zscore(spread)
    signal_series = cal_signal(zscore_series, 2, -2)
    position = cal_position(signal_series)
    df = pd.DataFrame({
        'zscore': zscore_series,
        'signal': signal_series,
        'pos': position
    })
    # 打印结果
    print(df)
