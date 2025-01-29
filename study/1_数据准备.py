import joblib
import pandas as pd

import config


def cut(candle, start, end):
    candle = candle[candle['candle_begin_time'] > start]
    candle = candle[candle['candle_begin_time'] < end]
    return candle


if __name__ == '__main__':
    data = pd.read_pickle(config.swap_path)
    start_date = '2023-12-01'
    end_date = '2025-01-01'
    btc = cut(data['BTC-USDT'], start_date, end_date)
    sample_data = {'BTC-USDT': btc}

    for symbol, df in data.items():
        tmp = cut(df, start_date, end_date)
        # 只保留了全年有数据的
        if len(tmp) == len(btc):
            sample_data[symbol] = tmp

    joblib.dump(sample_data, 'data2024.pkl')
