"""
因子选取：size ('size') & book-to-market ('btm')

1. 选因子值总和最大的股票（负相关因子值取负后求和）

具体做法：
- 每月最后一个交易日：按股票因子值总和（- size + btm）降序排序（因子均已标准化）
- 次月第一个交易日：开盘时，购买上月底因子值总和最大的10只股票，并持有到下个月第一个交易日开盘时（如果股票仍在股票池则连续持有）
- 每日计算日收益用第二天开盘价减去当天开盘价

2. 选预期收益最大的股票

具体做法：
- 用过去一段时间拟合股票的收益率和各因子间的关系（本策略仅考虑线性回归）
- 每月最后一个交易日：预测股票未来收益率并降序排序
- 次月第一个交易日：开盘时，购买上月预测收益率最大的10只股票，并持有到下个月第一个交易日开盘时（如果股票仍在股票池则连续持有）
- 每日计算日收益用第二天开盘价减去当天开盘价
"""

from FFM_test_new import *
import matplotlib
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def end_trade_date_each_month(start_date, end_date):
    """
    每月最后一个交易日列表（从持仓开始日期前一个交易日至持仓结束日期）；
    如持仓20040901-20181031，则返回数列第一个元素为'20040831'（使2004年9月能够持仓），最后一个元素为'20181031'。
    :param start_date: 持仓开始日期（月初）
    :param end_date: 持仓结束日期（月末）
    :return: 每月最后一个交易日列表（从持仓开始日期前一个交易日至持仓结束日期）
    """
    start_yr = int(start_date[:4])
    start_mo = int(start_date[4:6])
    end_yr = int(end_date[:4])
    end_mo = int(end_date[4:6])

    if start_mo == 1:
        start_mo_sort = 12
        start_year_sort = start_yr - 1
    else:
        start_mo_sort = start_mo - 1
        start_year_sort = start_yr

    end_trade_date_each_month_lst = create_test_dates_list(start_year_sort, start_mo_sort, end_yr, end_mo)
    return end_trade_date_each_month_lst


def trade_dates_each_month(end_date, next_end_date):
    trade_dates_df = pro.trade_cal(start_date=end_date, end_date=next_end_date)
    trade_dates_df = trade_dates_df[trade_dates_df.is_open == 1]
    trade_dates = list(trade_dates_df['cal_date'])[1:]
    return trade_dates


def get_k_data_from_csv(date):
    """
    从csv文件调取该交易日所有股票的日线数据。
    :param date: 日期（str）
    :return: 该交易日所有股票的日线数据（DataFrame）
    """
    path = '/Users/yanxinyu/Desktop/fama-french-validation/data/K_data_by_date/K_{}.csv'.format(date)
    try:
        k = pd.DataFrame(pd.read_csv(path, index_col=0))
    except FileNotFoundError:
        k = pro.daily(trade_date=date)
        k.set_index('ts_code', inplace=True)
        k.sort_values(by='ts_code', inplace=True)
        print("Writing new k data ({})...".format(date))
        k.to_csv(path)
        print("--- OK! ---")
    return k


"""
******************************选因子值总和最大的股票(max factors' sum)***************************************
"""


def sort_stocks_by_mfs(date):
    """
    月末计算所有股票因子值总和并降序排序。
    :param date: 每月最后一个交易日（string）
    :return: 排序后的股票列表（list）
    """

    """
    ******************************数据获取***********************************************************
    """
    # 月末所有股票代码列表（上市六个月以上非金融公司）
    stock_pool = [stock for stock in all_stocks_ts_code if stock_pool_filter(stock, date) is True]

    # 从csv文件获取该天所有股票的基本面数据
    daily_basic_df = get_daily_basic_data_from_csv(date)

    # 将所需数据放入新的DataFrame（factor_df）
    factor_df = pd.DataFrame(columns=['size', 'btm'])
    for stock in stock_pool:
        try:
            total_mv, pb = daily_basic_df.loc[stock, ['total_mv', 'pb']]
            size = np.log(total_mv)
            book_to_market = 1 / pb

            factor_df.loc[stock] = [size, book_to_market]
        # 如果该股票没有基本面数据则跳过
        except KeyError:
            continue

    """
    ******************************数据预处理**********************************************************
    """
    # 1. 离群值处理（MAD法）+ 2. 数据标准化
    factor_df['standard_size'] = standardize(remove_abnormal(factor_df['size'], 10))
    factor_df['standard_btm'] = standardize(remove_abnormal(factor_df['btm'], 10))

    # 3. 缺失值处理
    factor_df.dropna(inplace=True)

    # 因子值加总
    factor_df['sum'] = - factor_df['standard_size'] + factor_df['standard_btm']

    # 按因子值总和排序
    factor_df.sort_values(by='sum', ascending=False, inplace=True)
    sorted_stocks_lst = factor_df.index
    return sorted_stocks_lst


def selected_stocks_by_mfs(end_date, next_end_date):
    """
    选出每个月持仓的10只股票。
    :param end_date: 持仓月前一个交易日（用来给股票排序）
    :param next_end_date: 持仓月最后一个交易日（定位月末交易日）
    :return: 本月持仓的股票代码列表
    """

    # 每月最后一个交易日根据股票因子值的大小排序
    sorted_stocks = sort_stocks_by_mfs(end_date)

    # 次月第一个交易日根据股票排序情况购买前10只股票，
    # 为了防止出现下个交易日某些股票买不到的情况，确保股票在下一个交易日交易
    month_first_trade_date = trade_dates_each_month(end_date, next_end_date)[0]
    selected_stocks = [stock for stock in sorted_stocks
                       if stock in get_k_data_from_csv(month_first_trade_date).index][:10]
    selected_stocks.sort()
    return selected_stocks


def portfolio_return_each_month(end_date, next_end_date):
    selected_stocks = selected_stocks_by_mfs(end_date, next_end_date)
    print(selected_stocks)
    month_trade_dates = trade_dates_each_month(end_date, next_end_date)
    month_portfolio_df = pd.DataFrame(index=month_trade_dates, columns=selected_stocks)

    for trade_date in month_trade_dates[:]:
        print(trade_date)
        # 获取日线数据，计算购买股票的平均开盘价和全市场股票平均开盘价
        k_df = get_k_data_from_csv(trade_date)
        pct_change_lst = []
        for stock in selected_stocks:
            try:
                pct_change = k_df.loc[stock, 'pct_change']/100
            except KeyError:
                pct_change = 0
            pct_change_lst.append(pct_change)
            print(pct_change)
        month_portfolio_df.loc[trade_date] = pct_change_lst

    month_portfolio_df['portfolio_return'] = month_portfolio_df.sum(axis=1) * 0.1
    path = '/Users/yanxinyu/Desktop/fama-french-validation/strategy_record/max_factors_sum/month_portfolio_{}.csv' \
        .format(end_date)
    month_portfolio_df.to_csv(path)

    return month_portfolio_df['portfolio_return']


def get_portfolio_return(start_date, end_date):

    end_trade_dates = end_trade_date_each_month(start_date, end_date)
    month_portfolio_return_lst = []
    for d1, d2 in zip(end_trade_dates[:-1], end_trade_dates[1:]):
        path = '/Users/yanxinyu/Desktop/fama-french-validation/strategy_record/max_factors_sum/month_portfolio_{}.csv'\
            .format(d1)
        try:
            month_portfolio = pd.DataFrame(pd.read_csv(path, index_col=0))['portfolio_return']
        except FileNotFoundError:
            month_portfolio = portfolio_return_each_month(d1, d2)
        month_portfolio_return_lst.append(month_portfolio)

    whole_portfolio_return = pd.concat(month_portfolio_return_lst)
    return whole_portfolio_return


def get_benchmark_return(benchmark, start_date, end_date):
    df = pro.index_daily(ts_code=benchmark, start_date=start_date, end_date=end_date)
    df.set_index('trade_date', inplace=True)
    df.sort_index(ascending=True, inplace=True)
    df['pct_change'] = df['close']/df['close'].shift(1) - 1
    return df['pct_change']


def plot_strategy_max_factors_sum(start_date, end_date):
    pass


if __name__ == "__main__":

    # 持仓时间：2005年1月-2009年12月
    # 股票排序时间：2004年12月-2009年11月，每月最后一个交易日
    # 股票换仓时间：2005年1月-2009年12月，每月第一个交易日
    START = '20040901'
    END = '20181031'

    portfolio_return_daily = get_portfolio_return('20140101', '20151231')
    portfolio_return_daily.index = pd.to_datetime(portfolio_return_daily.index, format='%Y%m%d', errors='ignore')
    strategy_return = (portfolio_return_daily+1).cumprod()
    # print(strategy_return)
    benchmark_return_daily = get_benchmark_return('399300.SZ', '20140101', '20151231')
    benchmark_return_daily.index = pd.to_datetime(benchmark_return_daily.index, format='%Y%m%d', errors='ignore')
    benchmark_return = (benchmark_return_daily+1).cumprod()
    # print(benchmark_return)
    return_data_daily = pd.concat([portfolio_return_daily, benchmark_return_daily], axis=1)
    return_data_daily.columns = ['strategy_return_daily', 'benchmark_return_daily']
    return_data_daily.dropna(inplace=True)

    excess_return_daily = return_data_daily['strategy_return_daily'] - return_data_daily['benchmark_return_daily']
    return_data_daily['excess_return'] = (excess_return_daily + 1).cumprod()

    return_data = pd.concat([strategy_return, benchmark_return], axis=1)
    return_data.columns = ['strategy_return', 'benchmark_return']
    return_data.dropna(inplace=True)
    print(return_data)

    x = pd.to_datetime(return_data.index, format='%Y%m%d', errors='ignore')
    plt.plot(x, return_data['strategy_return'], label='strategy_return')
    plt.plot(x, return_data['benchmark_return'], label='benchmark_return')
    # plt.plot(x, return_data_daily['excess_return'], label='excess_return')
    plt.legend()
    plt.show()
