import pandas as pd

from functions import *


def plot_coint_pair(all_df: Dict[str, pd.DataFrame], symbol1: str, symbol2: str, start_date, end_date):
    base = extract_col(all_df[symbol1], col='close', start_time=start_date, end_time=end_date)
    target = extract_col(all_df[symbol2], col='close', start_time=start_date, end_time=end_date)

    summary = cal_cointegration(base, target)
    spread = cal_spread(base, target, summary['hedge_ratio'])
    zscore = cal_zscore(spread)
    pass
