import config
from program.analyse import cal_cointegration
from program.common import extract_col, cal_spread
from program.function import *

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
    signal = cal_signal(zscore_series, 2, -2)
    position = cal_position(signal)
    invert = invert_position(position)
    # 打印结果
    signal['pos'] = position
    signal['invert'] = invert
    print(signal)
