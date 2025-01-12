import itertools
import os

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def cal_evaluate(df: pd.DataFrame):
    """
    F神（FreeStep）策略评价函数
    :param df: 净值曲线
    :return: 评价
    """
    # 计算统计指标
    key = '策略评价'
    eps = 1e-9
    results = pd.DataFrame()
    ls_df = df.copy()
    ls_df.set_index('candle_begin_time', inplace=True)
    curve = ls_df['equity_curve'].to_frame(key)
    curve.index.name = 'candle_begin_time'
    time_diff = curve.index[-1] - curve.index[0]
    hour_diff = time_diff.total_seconds() / 3600
    curve_ = curve.copy()
    curve.reset_index(inplace=True)
    curve['本周期多空涨跌幅'] = curve[key].pct_change().fillna(0)
    # 累积净值
    results.loc[key, '累积净值'] = round(curve[key].iloc[-1], 3)
    # 计算当日之前的资金曲线的最高点
    curve['max2here'] = curve[key].expanding().max()
    # 计算到历史最高值到当日的跌幅,drowdwon
    curve['dd2here'] = curve[key] / curve['max2here'] - 1
    # 计算最大回撤,以及最大回撤结束时间
    end_date, max_draw_down = tuple(curve.sort_values(by=['dd2here']).iloc[0][['candle_begin_time', 'dd2here']])
    # 计算最大回撤开始时间
    start_date = curve[curve['candle_begin_time'] <= end_date].sort_values(by=key, ascending=False).iloc[0][
        'candle_begin_time']
    # 将无关的变量删除
    curve.drop(['max2here', 'dd2here'], axis=1, inplace=True)
    results.loc[key, '最大回撤'] = format(max_draw_down, '.2%')
    results.loc[key, '最大回撤开始时间'] = str(start_date)
    results.loc[key, '最大回撤结束时间'] = str(end_date)
    # ===统计每个周期
    results.loc[key, '盈利周期数'] = len(curve.loc[curve['本周期多空涨跌幅'] > 0])  # 盈利笔数
    results.loc[key, '亏损周期数'] = len(curve.loc[curve['本周期多空涨跌幅'] <= 0])  # 亏损笔数
    results.loc[key, '胜率'] = format(results.loc[key, '盈利周期数'] / (len(curve) + eps), '.2%')  # 胜率
    results.loc[key, '每周期平均收益'] = format(curve['本周期多空涨跌幅'].mean(), '.3%')  # 每笔交易平均盈亏
    if curve.loc[curve['本周期多空涨跌幅'] <= 0]['本周期多空涨跌幅'].mean() != 0:
        results.loc[key, '盈亏收益比'] = round(
            curve.loc[curve['本周期多空涨跌幅'] > 0]['本周期多空涨跌幅'].mean() / curve
            .loc[curve['本周期多空涨跌幅'] <= 0]['本周期多空涨跌幅'].mean() * (-1), 2)  # 盈亏比
    else:
        results.loc[key, '盈亏收益比'] = np.nan
    results.loc[key, '单周期最大盈利'] = format(curve['本周期多空涨跌幅'].max(), '.2%')  # 单笔最大盈利
    results.loc[key, '单周期大亏损'] = format(curve['本周期多空涨跌幅'].min(), '.2%')  # 单笔最大亏损
    # ===连续盈利亏损
    results.loc[key, '最大连续盈利周期数'] = max(
        [len(list(v)) for k, v in itertools.groupby(np.where(curve['本周期多空涨跌幅'] > 0, 1, np.nan))])  # 最大连续盈利次数
    results.loc[key, '最大连续亏损周期数'] = max(
        [len(list(v)) for k, v in itertools.groupby(np.where(curve['本周期多空涨跌幅'] <= 0, 1, np.nan))])  # 最大连续亏损次数
    # ===每年、每月收益率
    curve.set_index('candle_begin_time', inplace=True)

    # 计算相对年化 最大回撤 信息系数 波动率
    result_stats = pd.DataFrame(index=['年化收益', '月化收益', '月信息比', '月化波动'], columns=curve_.columns)

    result_stats.loc['年化收益'] = np.power(curve_.iloc[-1], 365 * 24 / hour_diff) - 1
    result_stats.loc['月化收益'] = np.power(curve_.iloc[-1], 30.4 * 24 / hour_diff) - 1
    result_stats.loc['月化波动'] = curve_.pct_change().dropna().apply(lambda x: x.std() * np.sqrt(30.5 * 24))
    result_stats.loc['月信息比'] = (result_stats.loc['月化收益'] / (result_stats.loc['月化波动'] + eps))
    result_stats = result_stats.astype('float32').round(3)

    data = multi_list_merge([result_stats.T, results])
    data['月化收益回撤比'] = data['月化收益'] / (abs(data['最大回撤'].str[:-1].astype('float32')) + eps) * 100

    data = data[['累积净值', '年化收益', '月化收益', '月信息比', '月化波动', '月化收益回撤比', '最大回撤',
                 '最大回撤开始时间', '最大回撤结束时间', '盈利周期数',
                 '亏损周期数', '胜率', '每周期平均收益', '盈亏收益比', '单周期最大盈利', '单周期大亏损',
                 '最大连续盈利周期数',
                 '最大连续亏损周期数']]
    data.to_csv('data.csv')
    return data


def multi_list_merge(df_list):
    merge_df = None
    for i in range(len(df_list) - 1):
        if i == 0:
            merge_df = pd.merge(df_list[0], df_list[1], left_index=True, right_index=True, how='inner')
        else:
            merge_df = merge_df.merge(df_list[i + 1], left_index=True, right_index=True, how='inner')
    return merge_df


def plot_output(x, data, data_path, save_html=True):
    x = x.copy()
    print(x)
    data.index.name = ''
    data = data[['累积净值', '年化收益', '月化收益', '月信息比', '月化波动', '月化收益回撤比', '累积净值',
                 '最大回撤', '最大回撤开始时间',
                 '最大回撤结束时间', '胜率', '盈亏收益比', '单周期最大盈利',
                 '单周期大亏损']].reset_index()
    part1 = data.iloc[:, :1].T.values.tolist()

    part2 = np.round(data.iloc[:, 1:7].T.values, 3).tolist()

    part3 = data.iloc[:, 7:].T.values.tolist()

    values = part1 + part2 + part3
    x['net_value'] = x['equity_curve'].round(4)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        specs=[[{"type": "table", "secondary_y": False}],
               [{"type": "xy", "secondary_y": True}],
               [{"type": "xy", "secondary_y": True}]],
        row_heights=[0.1, 0.75, 0.15],
    )

    # 主图
    fig.add_trace(
        go.Scatter(x=x['candle_begin_time'], y=x['net_value'], mode='lines', name='策略净值'),
        secondary_y=False, row=2, col=1,
    )

    fig.add_trace(
        go.Scatter(x=x['candle_begin_time'], y=(x['net_value'] / x['net_value'].cummax() - 1).round(4), mode='lines',
                   name='最大回撤',
                   line={'color': 'rgba(192,192,192,0.6)', 'width': 1}),
        secondary_y=True, row=2, col=1,
    )

    # 副图
    fig.add_trace(
        go.Table(
            header=dict(values=list(data.columns),  # 表头取值是data列属性
                        align='center'),
            cells=dict(values=values,  # 单元格的取值就是每个列属性的Series取值
                       fill_color='lavender',
                       align='center'),
            columnwidth=[40, 40, 40, 40, 40, 40, 60, 40, 40, 90, 90, 40, 40, 60, 60]),
        secondary_y=False, row=1, col=1,
    )
    fig.update_layout(
        yaxis_type='log', yaxis2_type='linear',
        template='none', hovermode='x', width=1650, height=950,
        xaxis_rangeslider_visible=False,
    )
    html_path = os.path.join(data_path, '净值曲线持仓图.html')

    if save_html:
        fig.write_html(file=html_path, config={'scrollZoom': True})

    fig.show(config={'scrollZoom': True})
