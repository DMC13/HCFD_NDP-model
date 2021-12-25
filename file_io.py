"""
HCFD Night Duty Planning Model
The file i/o functions.

Copyright (c) 2021 Tao-Ming Chen.
Licensed under the LGPL-2.1 License (see LICENSE for details)
"""

import os
import pandas as pd

import utils

#################################
#   Environment Setup
#################################


def detect_colab():
    # Import colab packages
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False
    return IN_COLAB


#################################
#   Input
#################################


def input_file(filename=None):
    # Input file
    if filename is None:
        filename = input('請輸入 excel 檔案的相對路徑：')
    print("--> 您選擇的檔案是：", filename)
    if not os.path.isfile(filename):
        raise FileNotFoundError('\n\n查無此檔案： %s' % filename)
    return filename


#################################
#   Output
#################################


def solutions2excel(input, bests):
    saved_file = input.filename[:-5] + '_result.xlsx'
    with pd.ExcelWriter(saved_file) as writer:
        for i, a_best_optiaml in enumerate(bests.best_k):
            df_result, df_result_short = utils.solution_to_df(
                input, a_best_optiaml.solution[0], a_best_optiaml.solution[1])
            df_xlsx = df_result.iloc[:, 1:].copy()
            df_xlsx = df_xlsx.append(pd.Series(name='', dtype='boolean'))
            df_xlsx = df_xlsx.append(df_result_short.iloc[:, :])
            df_xlsx.to_excel(writer, sheet_name=str(i+1))
    print('\n已將規劃結果儲存存於：', saved_file)
    return saved_file


if __name__ == '__main__':
    IN_COLAB = detect_colab()
    not_string = 'NOT '
    print(
        f'this file is {not_string if not IN_COLAB else ""}running in Colab.')
