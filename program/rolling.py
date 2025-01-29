import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


def cal_coint(base: pd.Series, target: pd.Series):
    """
    币对协整性判断
    :param base: x
    :param target: y
    :return: true/false
    """
    # 回归分析
    # 添加常数项
    X = sm.add_constant(base)
    # 拟合线性回归模型
    model = sm.OLS(target, X).fit()
    # 获取残差
    residuals = model.resid
    # adf检测
    adf_result = adfuller(residuals, autolag='AIC')
    p_value = adf_result[1]
    is_coint = p_value < 0.05 and adf_result[0] < adf_result[4]['5%']
    h_ratio = round(model.params.iloc[1], 5)
    return is_coint, h_ratio


def rolling_coint(s1: pd.Series, s2: pd.Series, back_hour=24 * 30):
    """
    滚动计算协整、对冲比率、相关系数，并确保索引与s1对齐
    """
    # 确保s1和s2索引一致
    s1_aligned, s2_aligned = s1.align(s2, join='inner')

    # 初始化结果列表
    coint_list = []
    h_ratio_list = []
    corr_list = []

    # 滚动窗口计算
    for i in range(back_hour, len(s1_aligned)):
        # 取窗口内的数据（严格历史数据）
        base = s1_aligned.iloc[i - back_hour:i]
        target = s2_aligned.iloc[i - back_hour:i]

        # 计算协整关系
        is_coint, h_ratio = cal_coint(base, target)
        corr = base.corr(target, method="spearman")

        # 保存结果
        coint_list.append(is_coint)
        h_ratio_list.append(h_ratio)
        corr_list.append(corr)

    # 生成结果Series，索引对齐s1的尾部
    result_index = s1_aligned.index[back_hour:]
    coint_series = pd.Series(coint_list, index=result_index, name='is_coint')
    h_ratio_series = pd.Series(h_ratio_list, index=result_index, name='hedge_ratio')
    corr_series = pd.Series(corr_list, index=result_index, name='corr')

    return coint_series, h_ratio_series, corr_series