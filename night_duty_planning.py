"""
HCFD Night Duty Planning Model
A sample of model implementation.

Copyright (c) 2021 Tao-Ming Chen.
Licensed under the LGPL-2.1 License (see LICENSE for details)
"""

import os
import numpy as np
import pandas as pd

import utils
import config
import file_io as fio
import model as modellib


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Initialize the NDP model.')
    parser.add_argument('--file_path', required=True,
                        metavar="/path/to/excel_file/filename.xlsx",
                        help='Path to the input excel file')
    parser.add_argument('--result_size', required=False,
                        metavar="5",
                        help='How many best optimals to record')
    parser.add_argument('--max_iter', required=False,
                        metavar="300",
                        help='Maximum number of iterations of the solver')
    parser.add_argument('--early_stop', required=False,
                        metavar="80",
                        help='Stop the solver when that many times the best optimal have not been changed')
    parser.add_argument('--from_main', required=False,
                        metavar="True", default=True,
                        help='Indicator showing that the script is running from terminal')
    args = parser.parse_args()


#################################
#   Config
#################################

class NDPconfig(config.Config):
    """
    Configurations for the NDP model.
    Derives from the base Config class and overrides some values.
    """
    if __name__ == '__main__':
        if args.result_size:
            result_size = int(args.result_size)
        if args.max_iter:
            max_iter = int(args.max_iter)
        if args.early_stop:
            early_stopping = int(args.early_stop)
        if args.from_main:
            from_main = True
    else:
        result_size = 5
        max_iter = 300
        early_stopping = 80


# Input File
IN_COLAB = fio.detect_colab()
input_file = args.file_path if args.file_path else None
filename = fio.input_file(filename=input_file)

# Prepare Model Input
x_table, y, members_info, date_info = utils.prepare_input(filename)
Input = modellib.NDPInput(filename, x_table, y, members_info, date_info)
conf = NDPconfig()

# Build and Solve the Model
model = modellib.NDPModel(Input, conf)
model.solve()

# Output File
saved_file = fio.solutions2excel(Input, model.bests)
