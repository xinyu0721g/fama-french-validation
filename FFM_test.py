"""
测试时间：2005年1月-2018年8月
选股范围：每月选取上市6个月以上的股票
基准指数：
"""

import tushare as ts
import pandas as pd
import numpy as np


def get_bs_data_from_tusharepro(ts_code_lst):
    """
    从tusharePro获取所有上市公司资产负债表，并保存为csv文件（BS_data文件夹）
    :param ts_code_lst: 股票代码列表
    :return: None
    """
    for ts_code in ts_code_lst:
        print(ts_code)
        bs_df = pro.balancesheet(ts_code=ts_code)
        bs_df.set_index('end_date', inplace=True)
        bs_df.sort_index(inplace=True)
        print(bs_df)
        path_base = '/Users/yanxinyu/Desktop/fama-french-validation'
        path = path_base + '/BS_data/BS_{}.csv'.format(ts_code)
        bs_df.to_csv(path)


def get_k_data_from_tusharepro(ts_code_lst):
    """
    从tusharePro获取所有上市公司日线行情，并保存为csv文件（K_data文件夹）
    :param ts_code_lst: 股票代码列表
    :return: None
    """
    for i in ts_code_lst:
        print(i)
        k_df = pro.daily(ts_code=i)
        k_df.set_index('trade_date', inplace=True)
        k_df.sort_index(inplace=True)
        path_base = '/Users/yanxinyu/Desktop/fama-french-validation'
        path = path_base + '/K_data/K_{}.csv'.format(i)
        k_df.to_csv(path)


def get_bs_data_from_csv(ts_code, test_date):
    """
    从csv文件读取该上市公司资产负债表数据
    :param ts_code: 股票代码
    :param test_date: 测试日期
    :return: total_share, total_assets, total_liab 或 None
    """
    path_bs = '/Users/yanxinyu/Desktop/fama-french-validation/BS_data/BS_{}.csv'.format(ts_code)
    bs_csv_data = pd.read_csv(path_bs, low_memory=False)
    bs_df = pd.DataFrame(bs_csv_data)
    bs_df.end_date = pd.to_datetime(bs_df.end_date, format='%Y%m%d', errors='ignore')
    try:
        for index in range(0, len(bs_df)):
            end_date = bs_df.loc[index, 'end_date']
            if end_date > test_date:
                break
        total_share, total_assets, total_liab = bs_df.loc[index - 1, ['total_share', 'total_assets', 'total_liab']]
        return total_share, total_assets, total_liab
    except KeyError:
        return None


def create_test_dates_list(start_yr, start_mo, end_yr, end_mo):
    """
    创建测试日期列表（测试期内每月月底）
    :param start_yr: 开始年份
    :param start_mo: 开始月份
    :param end_yr: 结束年份
    :param end_mo: 结束月份
    :return: 测试日期列表
    """
    test_dates_lst_local = []
    for yr in range(start_yr, end_yr+1):
        if yr == start_yr:
            month_range = range(start_mo, 13)
        elif yr == end_yr:
            month_range = range(1, end_mo)
        else:
            month_range = range(1, 13)

        for mo in month_range:
            if mo != 12:
                test_date = (pd.datetime(yr, mo+1, 1) - pd.to_timedelta(1, unit='days'))
            else:
                test_date = pd.datetime(yr, mo, 31)
            test_dates_lst_local.append(test_date)

    return test_dates_lst_local


def stock_pool_filter(ts_code, test_date):
    """
    每月选取符合条件的股票放入股票池（上市且六个月以上）
    :param ts_code: 股票代码
    :param test_date: 建立股票池的日期（每月底）
    :return: 布尔值，确定股票是否进入股票池
    """
    list_date_i = stock_pool_all.loc[ts_code, 'list_date']
    delist_date_i = stock_pool_all.loc[ts_code, 'delist_date']
    listing_days = (test_date - list_date_i).days

    # 如果该股票退市，判断建立股票池时该股票上市时间是否还剩余一年时间，如果不是则不入股票池
    if delist_date_i is not None:
        remaining_days = (delist_date_i - test_date).days
        if remaining_days < 365:
            return False
        # 如果剩余时间超过一年，则上市时间超过半年
        else:
            if listing_days >= 183:
                return True
            else:
                return False
    # 如果该股票没有退市，则只需判断建立股票池时股票上市是否满半年
    else:
        if listing_days >= 183:
            return True
        else:
            return False


if __name__ == "__main__":

    pro = ts.pro_api()

    # 获取所有股票列表
    stock_L = pro.stock_basic(list_status='L', fields='ts_code,name,industry,list_date,delist_date')
    stock_D = pro.stock_basic(list_status='D', fields='ts_code,name,industry,list_date,delist_date')
    stock_P = pro.stock_basic(list_status='P', fields='ts_code,name,industry,list_date,delist_date')
    stock_pool_all = stock_L.append(stock_D).append(stock_P)

    # 剔除金融类公司，剩余3586只股票
    stock_pool_all = stock_pool_all[stock_pool_all.industry != '银行']
    stock_pool_all = stock_pool_all[stock_pool_all.industry != '保险']
    stock_pool_all = stock_pool_all[stock_pool_all.industry != '证券']

    # 设置股票代码为索引，将日期转为datetime类型
    stock_pool_all.set_index('ts_code', inplace=True)
    stock_pool_all.list_date = pd.to_datetime(stock_pool_all.list_date, format='%Y%m%d', errors='ignore')
    stock_pool_all.delist_date = pd.to_datetime(stock_pool_all.delist_date, format='%Y%m%d', errors='ignore')
    # print(stock_pool_all)
    stock_code_lst = list(stock_pool_all.index)
    # print(stock_code_lst)

    # 测试时间：2005年1月-2018年8月
    test_dates_lst = create_test_dates_list(2005, 1, 2018, 9)

    # 测试：回归第一个月,return～size for all stock
    test_date = test_dates_lst[0]
    stock_lst = []
    for i in stock_code_lst:
        if stock_pool_filter(i, test_date) is True:
            stock_lst.append(i)
    print(len(stock_lst))
    test_date_str = str(test_date)[:10].replace('-', '')
    print(test_date_str)

    day1 = pro.daily_basic(ts_code='', trade_date=test_date_str)
    print(day1)

    # effect_stock = []
    # size_lst = []
    # return_lst = []
    # for i in stock_lst:
    #
    #     if get_bs_data_from_csv(i, test_date) is not None:
    #         effect_stock.append(i)
    #         total_share, total_assets, total_liab = get_bs_data_from_csv(i)

