"""
测试时间：2005年1月-2018年8月
选股范围：每月选取上市6个月以上的股票（剔除金融类公司）
基准指数：全市场按市值加权总收益

数据获取：
1. 每个月最后一个交易日：获取该天所有符合条件的上市公司的基本面信息和收盘价；
2. 下个月最后一个交易日：获取该天上述公司的收盘价，计算monthly_return；
3. 将因子和收益放入新的DataFrame

数据预处理：
1. 离群值处理
2. 数据标准化
3. 缺失值处理

单因子测试：
1. 稳健回归
2. 分组回测
"""

import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import csv
import os


def get_all_stocks():
    """
    得到所有上市公司的股票列表（剔除金融类公司）。
    :return: 所有上市公司的股票列表（DataFrame）
    """

    # 获取所有股票列表
    stock_pool = pro.stock_basic(list_status='L', fields='ts_code,name,industry,list_date')

    # 剔除金融类公司（银行、保险、证券）
    stock_pool = stock_pool[stock_pool.industry != '银行']
    stock_pool = stock_pool[stock_pool.industry != '保险']
    stock_pool = stock_pool[stock_pool.industry != '证券']

    # 将股票代码设为索引，将上市和退市日期转为datetime类型
    stock_pool.set_index('ts_code', inplace=True)
    stock_pool.list_date = pd.to_datetime(stock_pool.list_date, format='%Y%m%d', errors='ignore')

    return stock_pool


def create_test_dates_list(start_yr, start_mo, end_yr, end_mo):
    """
    获取每个月最后一个交易日的列表。
    :param start_yr: 起始年（int）
    :param start_mo: 起始年的起始月（int）
    :param end_yr: 结束年（int）
    :param end_mo: 结束年的结束月（int）
    :return: 每个月最后一个交易日的列表（list）
    """

    # 开始日为开始年开始月的第一天，结束日为结束年结束月下一月的第一天（确保能取到结束月的最后一天）
    if start_mo in [10, 11, 12]:
        start_date = '{}{}{}'.format(start_yr, start_mo, '01')
    else:
        start_date = '{}{}{}'.format(start_yr, '0' + str(start_mo), '01')

    if end_mo == 12:
        end_date = '{}{}{}'.format(end_yr + 1, '01', '01')
    elif end_mo in [9, 10, 11]:
        end_date = '{}{}{}'.format(end_yr, end_mo + 1, '01')
    else:
        end_date = '{}{}{}'.format(end_yr, '0' + str(end_mo + 1), '01')

    # 从tusharePro获取该时间段所有交易日列表
    trade_dates_df = pro.trade_cal(start_date=start_date, end_date=end_date)
    trade_dates_df = trade_dates_df[trade_dates_df.is_open == 1]  # is_open=1 表示该天交易

    # 每个月最后一个交易日的列表
    month_end_trade_dates = []
    for yr in range(start_yr, end_yr + 1):
        if start_yr == end_yr:
            month_lst = range(start_mo, end_mo + 1)
        elif yr == end_yr:
            month_lst = range(1, end_mo + 1)
        elif yr == start_yr:
            month_lst = range(start_mo, 13)
        else:
            month_lst = range(1, 13)

        for mo in month_lst:
            if mo in [10, 11, 12]:
                yr_mo = str(yr) + str(mo)
            else:
                yr_mo = str(yr) + '0' + str(mo)
            yr_mo_trade_dates_lst = [date for date in list(trade_dates_df.cal_date) if date.startswith(yr_mo)]
            yr_mo_last_trade_date = yr_mo_trade_dates_lst[-1]
            month_end_trade_dates.append(yr_mo_last_trade_date)

    return month_end_trade_dates


def get_daily_basic_data_from_csv(date):
    """
    从csv文件调取该交易日所有股票的基本面数据。
    :param date: 日期（str）
    :return: 该交易日所有股票的基本面数据（DataFrame）
    """
    path = '/Users/yanxinyu/Desktop/fama-french-validation/data/DailyBasic_data/DailyBasic_{}.csv'.format(date)
    try:
        dailybasic_df = pd.DataFrame(pd.read_csv(path, low_memory=False, index_col=0))
    except FileNotFoundError:
        dailybasic_df = pro.daily_basic(trade_date=date)
        dailybasic_df.set_index('ts_code', inplace=True)
        print("Writing new daily basic data ({})...".format(date))
        dailybasic_df.to_csv(path)
        print("--- OK! ---")
    return dailybasic_df


def stock_pool_filter(ts_code, test_date):
    """
    每月选取符合条件的股票放入股票池（上市六个月以上）
    :param ts_code: 股票代码
    :param test_date: 建立股票池的日期（每月最后一个交易日）
    :return: 布尔值，确定股票是否进入股票池
    """
    test_date = pd.to_datetime(test_date)
    list_date = all_stocks.loc[ts_code, 'list_date']
    listing_days = (test_date - list_date).days

    if listing_days >= 180:
        return True
    else:
        return False


def remove_abnormal(series, n):
    """
    用MAD方法处理离群值。
    :param series: 待处理数列（series，index为股票代码）
    :param n: MAD法参数
    :return: 离群值处理后的数列（series，index为股票代码）
    """

    # 第一步：找出所有因子的中位数Xmedian
    Xmedian = series.median()

    # 第二步：计算每个因子与中位数的绝对偏差值
    absolute_deviation_series = pd.Series([abs(X - Xmedian) for X in series])

    # 第三步：得到绝对偏差值的中位数MAD
    mad = absolute_deviation_series.median()

    # 第四步：确定因子合理范围[Xmedian-nMAD, Xmedian+nMAD]
    upper_bound = Xmedian + n * mad
    lower_bound = Xmedian - n * mad

    # 最后：调整因子，形成新列
    return np.clip(series, lower_bound, upper_bound)


def standardize(series):
    """
    数据z-score标准化处理。
    :param series: 待处理数列（series，index为股票代码）
    :return: 标准化处理后的数列（series，index为股票代码）
    """
    return (series - series.mean())/series.std()


def create_test_df(date, next_date):
    """
    将因子和收益放入新的DataFrame。
    :param date: 本月最后一个交易日（string）
    :param next_date: 次月最后一个交易日（string）
    :return: 用以稳健回归得因子和收益df（DataFrame）
    """

    """
    ******************************数据获取***********************************************************
    """
    # 本月所有股票代码列表（上市六个月以上非金融公司）
    stock_lst = [stock for stock in all_stocks_ts_code if stock_pool_filter(stock, date) is True]

    # 从csv文件获取本月和次月所有股票的基本面数据
    dailybasic_df = get_daily_basic_data_from_csv(date)
    dailybasic_df_next = get_daily_basic_data_from_csv(next_date)

    # 将所需数据放入新的DataFrame（test_df）
    test_df = pd.DataFrame(columns=['size', 'btm', 'ep', 'return'])
    for stock in stock_lst:
        try:
            total_mv, pb, pe, close = dailybasic_df.loc[stock, ['total_mv', 'pb', 'pe', 'close']]
            size = np.log(total_mv)
            book_to_market = 1 / pb
            ep = 1 / pe

            close_next = dailybasic_df_next.loc[stock, 'close']
            monthly_return = np.log(close_next / close)
            test_df.loc[stock] = [size, book_to_market, ep, monthly_return]

        # 如果本月或次月该股票不在股票池内则跳过（不是上市六个月以上非金融公司）
        except KeyError:
            continue

    # 计算市场收益率（按市值加权的全市场收益）
    test_df['weight'] = test_df['size'] / sum(test_df['size'])
    market_return = sum(test_df['weight'] * test_df['return'])
    test_df['excess_return'] = test_df['return'] - market_return

    """
    ******************************数据预处理**********************************************************
    """
    # 1. 离群值处理（MAD法）+ 2. 数据标准化
    test_df['standard_size'] = standardize(remove_abnormal(test_df['size'], 10))
    test_df['standard_btm'] = standardize(remove_abnormal(test_df['btm'], 10))
    test_df['standard_ep'] = standardize(remove_abnormal(test_df['ep'], 10))
    test_df['standard_ex_return'] = standardize(remove_abnormal(test_df['excess_return'], 10))

    # 3. 缺失值处理
    test_df.dropna(inplace=True)
    print(test_df)

    return test_df


"""
******************************稳健回归***********************************************************
"""


def fit_data_rlm(x, y):
    """
    稳健回归（Robust Regression）
    :param x: 自变量（series）
    :param y: 因变量（series）
    :return: 回归系数和p值
    """
    x = sm.add_constant(x)
    est = sm.RLM(y, x).fit()
    coefficient = round(est.params[1], 4)
    p_value = round(est.pvalues[1], 4)
    # print(est.summary())
    #
    # x_prime = np.linspace(x.iloc[:, 1].min(), x.iloc[:, 1].max(), 100)
    # x_prime = sm.add_constant(x_prime)
    # y_hat = est.predict(x_prime)
    # plt.scatter(x.iloc[:, 1], y, alpha=0.3)
    # plt.xlabel('factor')
    # plt.ylabel('monthly excess return')
    # plt.plot(x_prime[:, 1], y_hat, 'r', alpha=0.9)
    # plt.show()
    return coefficient, p_value


def init_rlm_test_csv(factor):
    """
    初始化单因子稳健回归测试结果的csv文件。
    :param factor: 因子名（string）
    :return: None
    """
    csv_path = '/Users/yanxinyu/Desktop/fama-french-validation/factor_test/rlm_test/{}_rlm_test.csv'.format(factor)
    with open(csv_path, 'w', newline='') as f:
        row = ['test_date', 'coefficient', 'p_value', 'significant', 'direction']
        out = csv.writer(f)
        out.writerow(row)
        f.close()


def new_row_rlm_test_to_csv(factor, test_df, date):
    """
    单因子稳健回归csv文件写入新的回归结果。
    :param factor: 因子名（string）
    :param test_df: 因子值和收益率序列的df（DataFrame）
    :param date: 测试日期（本月最后一个交易日）
    :return: None
    """
    csv_path = '/Users/yanxinyu/Desktop/fama-french-validation/factor_test/rlm_test/{}_rlm_test.csv'.format(factor)

    # 1.每月将全市场（上市6个月以上非金融公司）的股票的指标值序列与次月超额收益序列做稳健回归

    coefficient, p_value = fit_data_rlm(test_df['standard_{}'.format(factor)], test_df['standard_ex_return'])
    print(coefficient, p_value)

    # 2.显著性和方向性：检验回归系数是否显著为正或显著为负
    if p_value >= 0.05 or coefficient == 0:
        significant = False
        direction = None
    else:
        significant = True
        if coefficient > 0:
            direction = 1
        else:
            direction = -1

    # 将结果写入csv文件
    new_line = [date, coefficient, p_value, significant, direction]

    with open(csv_path, 'a', newline='') as f:
        out = csv.writer(f)
        out.writerow(new_line)
        f.close()


def single_factor_rlm_test_to_csv(factor):
    """
    单因子稳健回归，并将结果写入csv文件（rlm_test文件夹）
    :param factor: 因子名（string）
    :return: None
    """

    init_rlm_test_csv(factor)
    for date, next_date in zip(month_end_trade_dates_lst[:-1], month_end_trade_dates_lst[1:]):
        print(date, next_date)
        test_df = create_test_df(date, next_date)
        new_row_rlm_test_to_csv(factor, test_df, date)


def get_rlm_results_from_csv(factor):
    """
    从csv文件获取某因子稳健回归测试结果。
    :param factor: 因子名（string）
    :return: 某因子稳健回归测试结果（DataFrame）
    """
    csv_path = '/Users/yanxinyu/Desktop/fama-french-validation/factor_test/rlm_test/{}_rlm_test.csv'.format(factor)
    rlm_test_csv_data = pd.DataFrame(pd.read_csv(csv_path, low_memory=False))
    return rlm_test_csv_data


def print_rlm_results(factor):
    """
    计算单因子稳健回归得到的：正/负相关显著比例、同向/切换次数占比等指标值。
    :param factor: 因子名（string）
    :return: 单因子稳健回归得到的指标值
    """
    try:
        rlm_results = get_rlm_results_from_csv(factor)
    except FileNotFoundError:
        print("Writing new rlm test results for factor {}...".format(factor))
        single_factor_rlm_test_to_csv(factor)
        rlm_results = get_rlm_results_from_csv(factor)
    total_tests = len(rlm_results)
    rlm_results.dropna(inplace=True)
    rlm_results.reset_index(drop=True, inplace=True)

    # 3.风格持续：如果当月的回归系数与最近一次显著的系数正负号相同，则记为同向；如果方向发生了切换，则记为切换
    # 同向显著次数占比越大，说明指标的显著性和趋势性越强
    # 切换次数较多意味着该指标所代表的市场风格经常发生转换，或者说风格持续的时间较短
    for i in rlm_results.index[1:]:
        if rlm_results.loc[i - 1, 'direction'] == rlm_results.loc[i, 'direction']:
            rlm_results.loc[i, 'keep_or_reverse'] = 1
        else:
            rlm_results.loc[i, 'keep_or_reverse'] = -1

    # print(rlm_results)
    direction_count = dict(rlm_results['direction'].value_counts())
    keep_count = dict(rlm_results['keep_or_reverse'].value_counts())

    # 正/负相关显著比例
    positive_percent = round(direction_count[1.0] / total_tests, 4)
    negative_percent = round(direction_count[-1.0] / total_tests, 4)
    # 同向/状态切换显著次数占比
    keep_percent = round(keep_count[1.0] / total_tests, 4)
    reverse_percent = round(keep_count[-1.0] / total_tests, 4)
    # 显著比例较高的方向
    posi_minus_nega = positive_percent - negative_percent
    if posi_minus_nega >= 0:
        factor_direction = '+'
    else:
        factor_direction = '-'
    # abs(正-负)
    abs_posi_nega = round(abs(posi_minus_nega), 4)
    # 同向-切换
    keep_minus_reverse = round((keep_percent - reverse_percent), 4)

    print('\n{}单因子稳健回归结果：'.format(factor))
    print('正相关显著比例：\t\t{0:4.2%} | 负相关显著比例：\t\t{1:4.2%} | abs(正-负)：\t{2:4.2%} | 显著比例较高的方向：{3:2s}'
          .format(positive_percent, negative_percent, abs_posi_nega, factor_direction))
    print('同向显著次数占比：\t\t{0:4.2%} | 状态切换次数占比：\t{1:4.2%} | 同向-切换：\t\t{2:4.2%}'
          .format(keep_percent, reverse_percent, keep_minus_reverse))

    return positive_percent, negative_percent, keep_percent, reverse_percent, \
        factor_direction, abs_posi_nega, keep_minus_reverse


"""
******************************分组回测***********************************************************
"""


def split_to_5_groups(factor, df):
    """
    将个股按照某个因子指标值等分为5组。
    :param factor: 因子名（string）
    :param df: 因子值和收益率序列的df（DataFrame）
    :return: test_df按照某个因子指标值等分为的5个df（DataFrame）
    """

    df.sort_values(by=factor, ascending=True, inplace=True)
    df.reset_index(inplace=True)

    each_len = len(df[factor]) / 5
    group1 = df.iloc[:int(each_len), :].copy()
    group2 = df.iloc[int(each_len): int(each_len * 2), :].copy()
    group3 = df.iloc[int(each_len * 2): int(each_len * 3), :].copy()
    group4 = df.iloc[int(each_len * 3): int(each_len * 4), :].copy()
    group5 = df.iloc[int(each_len * 4):, :].copy()
    return group1, group2, group3, group4, group5


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        print("Making new folder...")
        os.makedirs(path)
        print("--- OK! ---")
    else:
        print("--- This folder exists! ---")


def init_group_test_csv(factor):
    """
    初始化分组回测结果的csv文件，每个因子分成5组。
    :param factor: 因子名（string）
    :return: None
    """
    csv_path_extend = ['/group{}.csv'.format(i) for i in range(1, 6)]
    csv_path_base = '/Users/yanxinyu/Desktop/fama-french-validation/factor_test/group_test/{}'.format(factor)
    mkdir(csv_path_base)
    csv_paths = [csv_path_base + extend for extend in csv_path_extend]

    for path in csv_paths:
        with open(path, 'w', newline='') as f:
            row = ['test_date', 'monthly_ex_return', 'tracking_error', 'IR']
            out = csv.writer(f)
            out.writerow(row)
            f.close()


def new_row_group_test_to_csv(extend, row):
    """
    分组回测结果的csv文件写入新的回测结果。
    :param extend: csv末路径+csv文件名（string）
    :param row: 新分组回测结果（list）
    :return: None
    """
    csv_path_base = '/Users/yanxinyu/Desktop/fama-french-validation/factor_test/group_test/'
    csv_path = csv_path_base + extend
    with open(csv_path, 'a', newline='') as f:
        out = csv.writer(f)
        out.writerow(row)
        f.close()


def single_factor_group_test_to_csv(factor):
    """
    单因子分组回测，并将结果写入csv文件（group_test文件夹）
    :param factor: 因子名（string）
    :return: None
    """

    init_group_test_csv(factor)

    # 做法：按照指标从小到大将市场分为5组，每月根据指标变化调整一次组合，然后计算各组相对于基准的超额收益和信息比率
    for date, next_date in zip(month_end_trade_dates_lst[:-1], month_end_trade_dates_lst[1:]):
        print(date, next_date)

        test_df = create_test_df(date, next_date)
        groups = list(split_to_5_groups(factor, test_df))
        for i in range(5):
            group = groups[i]
            csv_path_extend = '{}/group{}.csv'.format(factor, i + 1)

            monthly_ex_return = round(group['excess_return'].mean(), 4)
            tracking_error = round(group['excess_return'].std(), 4)
            info_ratio = round(monthly_ex_return / tracking_error, 4)

            row = [date, monthly_ex_return, tracking_error, info_ratio]
            new_row_group_test_to_csv(csv_path_extend, row)


def get_group_results_from_csv(factor):
    """
    从csv文件获取某因子分组回测结果。
    :param factor: 因子名（str）
    :return: 某因子稳健回归测试结果（元素为DataFrame的list）
    """
    csv_path_base = '/Users/yanxinyu/Desktop/fama-french-validation/factor_test/group_test/{}/'.format(factor)
    csv_paths = [csv_path_base + 'group{}.csv'.format(i) for i in range(1, 6)]
    group_results_all_dates = [pd.DataFrame(pd.read_csv(path, low_memory=False)) for path in csv_paths]
    return group_results_all_dates


def print_group_results(factor):

    print('\n{}因子分组回测结果：'.format(factor))
    try:
        group_results_all_dates = get_group_results_from_csv(factor)
    except FileNotFoundError:
        print("Writing new group test results for factor {}...".format(factor))
        single_factor_group_test_to_csv(factor)
        group_results_all_dates = get_group_results_from_csv(factor)

    group_test_result_df = pd.DataFrame(index=['group{}'.format(i) for i in range(1, 6)],
                                        columns=['monthly_ex_return', 'IR'])
    for i in range(1, 6):
        group_test_result_df.loc['group{}'.format(i), ['monthly_ex_return', 'IR']] \
            = round(group_results_all_dates[i - 1][['monthly_ex_return', 'IR']].mean(), 4)
    print(group_test_result_df)

    plt.bar(range(5), group_test_result_df['monthly_ex_return'], align='center', color='steelblue', alpha=0.8)
    plt.ylabel('monthly excess return')
    plt.title('{} factor group test'.format(factor))
    plt.xticks(range(5), group_test_result_df.index)
    plt.show()


"""
******************************因子间相关性及替代**************************************************
"""


def init_factor_corr_test_csv(f1, f2):
    """
    初始化因子分层组合测试结果的csv文件，按因子1分层后再按因子2分组。
    :param f1: 因子1（string）
    :param f2: 因子2（string）
    :return: None
    """
    csv_path_base = '/Users/yanxinyu/Desktop/fama-french-validation/factor_test/corr_test/{}-{}'.format(f1, f2)
    mkdir(csv_path_base)
    for group_i in range(1, 6):
        for subgroup_i in range(1, 6):
            csv_path_extend = '/{}{}-{}{}.csv'\
                .format(f1, group_i, f2, subgroup_i)
            path = csv_path_base + csv_path_extend
            with open(path, 'w', newline='') as f:
                row = ['test_date', 'monthly_ex_return', 'tracking_error', 'IR']
                out = csv.writer(f)
                out.writerow(row)
                f.close()


def new_row_corr_test_to_csv(extend, row):
    """
    分组回测结果的csv文件写入新的回测结果。
    :param extend: csv末路径+csv文件名（string）
    :param row: 新分组回测结果（list）
    :return: None
    """
    csv_path_base = '/Users/yanxinyu/Desktop/fama-french-validation/factor_test/corr_test/'
    csv_path = csv_path_base + extend
    with open(csv_path, 'a', newline='') as f:
        out = csv.writer(f)
        out.writerow(row)
        f.close()


def factor_corr_to_csv(f1, f2):
    """
    因子分层分组回测，并将结果写入csv文件（corr_test文件夹）
    :param f1: 控制因子（string）
    :param f2: 测试因子（string）
    :return: None
    """

    init_factor_corr_test_csv(f1, f2)
    for date, next_date in zip(month_end_trade_dates_lst[:-1], month_end_trade_dates_lst[1:]):
        print(date, next_date)

        test_df = create_test_df(date, next_date)
        groups = list(split_to_5_groups(f1, test_df))

        for group_i in range(5):
            group = groups[group_i]
            sub_groups = list(split_to_5_groups(f2, group))

            for subgroup_i in range(5):
                subgroup = sub_groups[subgroup_i]
                csv_path_extend = '{}-{}/{}{}-{}{}.csv' \
                    .format(f1, f2, f1, group_i + 1, f2, subgroup_i + 1)

                monthly_ex_return = round(subgroup['excess_return'].mean(), 4)
                tracking_error = round(subgroup['excess_return'].std(), 4)
                info_ratio = round(monthly_ex_return / tracking_error, 4)

                new_row = [date, monthly_ex_return, tracking_error, info_ratio]
                new_row_corr_test_to_csv(csv_path_extend, new_row)


def print_factor_corr_results(f1, f2):

    print('\n{}-{}因子分组分层回测结果：'.format(f1, f2))
    csv_path_base = '/Users/yanxinyu/Desktop/fama-french-validation/factor_test/corr_test/'
    corr_test_result_df = pd.DataFrame(index=['{}{}'.format(f1, group_i) for group_i in range(1, 6)],
                                       columns=['{}{}'.format(f2, subgroup_i) for subgroup_i in range(1, 6)])
    for group_i in range(1, 6):
        for subgroup_i in range(1, 6):
            path = csv_path_base + '{}-{}/{}{}-{}{}.csv'.format(f1, f2, f1, group_i, f2, subgroup_i)
            try:
                subgroup_return_df = pd.DataFrame(pd.read_csv(path, low_memory=False))
            except FileNotFoundError:
                print("Writing new factor correlation test for factor {} and factor {}...".format(f1, f2))
                factor_corr_to_csv(f1, f2)
                subgroup_return_df = pd.DataFrame(pd.read_csv(path, low_memory=False))
            corr_test_result_df.loc['{}{}'.format(f1, group_i), '{}{}'.format(f2, subgroup_i)] \
                = round(subgroup_return_df['monthly_ex_return'].mean(), 4)
    print(corr_test_result_df)


pro = ts.pro_api()

# 获取所有上市公司的股票代码
all_stocks = get_all_stocks()
all_stocks_ts_code = all_stocks.index


if __name__ == "__main__":

    # 测试时间：2005年1月-2018年8月（164个月）
    # 获取每个月最后一个交易日的列表（每个元素为日期string，如'20050131'）
    month_end_trade_dates_lst = create_test_dates_list(2005, 1, 2018, 9)
    print(month_end_trade_dates_lst)

    """
    ******************************单因子检验*******************************************************
    """
    factor1 = 'size'
    factor2 = 'btm'
    factor3 = 'ep'

    """
    ******************************稳健回归***************************
    """

    print_rlm_results(factor1)
    print_rlm_results(factor2)
    print_rlm_results(factor3)

    """
    ******************************分组回测***************************
    """

    # print_group_results(factor1)
    # print_group_results(factor2)
    # print_group_results(factor3)

    """
    ******************************因子相关性和替代**************************************************
    """
    """
    ******************************因子分层***************************
    """

    # print_factor_corr_results(factor1, factor2)
    # print_factor_corr_results(factor1, factor3)
    # print_factor_corr_results(factor2, factor3)
