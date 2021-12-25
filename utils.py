"""
HCFD Night Duty Planning Model
Common utility functions and classes.

Copyright (c) 2021 Tao-Ming Chen.
Licensed under the LGPL-2.1 License (see LICENSE for details)
"""

import copy
import numpy as np
import pandas as pd


def prepare_input(filename):
    """
    generate the required variables for a modellib.NDPInput object.
    modify this function if your excel data format 
    is different from mine.
    """

    try:
        df = pd.read_excel(filename, index_col=0)

        squad = tuple(df['班別'][1:])
        shift_off = tuple(df.loc['輪休'][1:-2])

        # 轉換資料：有服勤 -> 1 ；休假 -> 0
        df = df.iloc[1:, 1:]
        df.iloc[:, :-2] = df.iloc[:, :-2].notnull().astype('int')
        df.iloc[:, :-2] = df.iloc[:, :-2].replace(1, 2)
        df.iloc[:, :-2] = df.iloc[:, :-2].replace(0, 1)
        df.iloc[:, :-2] = df.iloc[:, :-2].replace(2, 0)

        x_table = df.iloc[:, :-2].to_numpy()
        y = df[['應值宿', '應救護']].astype('int').to_numpy()

        members_id = tuple(df.index)
        n_members = len(members_id)
        days = tuple(df.columns)[1:-2]
        n_days = len(days)

        # check if values are valid.
        check_data(members_id, days, n_days, y, squad, shift_off)

        members_info = (members_id, n_members, squad)
        date_info = (days, n_days, shift_off)

        return x_table, y, members_info, date_info

    except:
        raise TypeError('輸入的 Excel 檔案格式不符，請修正後重新執行程式')


def check_data(members_id, days, n_days, y, squad, shift_off):
    # check if values are integers.
    for value in (members_id+days):
        if not isinstance(value, int):
            raise ValueError('隊員的番號、日期應為整數，請修正後重新執行程式')

    # check if the values in '應值宿' & '應救護'
    # match the number of days of this month.
    sum_y = y.sum(axis=0)
    if sum_y[0] != n_days or sum_y[1] != n_days*2:
        raise ValueError('應值宿或應救護的次數與天數不符，請修正後重新執行程式')

    # check if the days are consective integers
    for i in range(n_days-1):
        if days[i]+1 != days[i+1]:
            raise ValueError('請將此月的每一天都列在 Excel 表中，請修正後重新執行程式')

    # check if the dayoff squad everyday is one of the squads.
    if not set(shift_off).issubset(set(squad)):
        raise ValueError('每日輪休的班別，與隊員的班別不一致，請修正後重新執行程式')


def solution_to_df(input, D, E):
    """
    based on an AnOptimal object, 
    generate output dataframe table.

    - input:    a modellib.NDPInput object
    - D:        model.bests.best_k[i].solution[0]
    - E:        model.bests.best_k[i].solution[1]
    """
    duty, night = list(), list()
    output_table = np.where(input.x_table > 0, 'v', '')

    for col, d in enumerate(input.days):
        for row, m in enumerate(input.members_id):
            if D[d, m].varValue > 0:
                output_table[row, col+1] = '宿'
                duty.append(m)
            elif E[d, m].varValue > 0:
                output_table[row, col+1] = '救'
                night.append(m)

    output_short = [duty]
    output_short.append([m for i, m in enumerate(night) if i % 2 == 0])
    output_short.append([m for i, m in enumerate(night) if i % 2 == 1])

    df_result = pd.DataFrame(
        data=output_table, columns=[0]+list(input.days), index=input.members_id)
    df_result['應值宿'] = input.y[:, 0]
    df_result['應救護'] = input.y[:, 1]
    df_result_short = pd.DataFrame(
        data=output_short, columns=input.days, index=['值宿', '夜1', '夜2'])

    return df_result, df_result_short
