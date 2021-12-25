"""
HCFD Night Duty Planning Model
Base Configurations class.

Copyright (c) 2021 Tao-Ming Chen.
Licensed under the LGPL-2.1 License (see LICENSE for details)
"""

# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.


class Config(object):

    # weight of variance(D) and variance(E)
    # given a set of var(D) and var(E),
    # total variance = weight[0] * var(D) + weight[1] * var(E)
    variance_weight = (1, 0)

    # 0 -> quiet; 1 -> basic info; 2 -> details (noisy)
    verbose = 1

    # score gainned respectively when assigning duties on
    # dayoff, rotate_day, first_day, second_day
    score = (-1000000, 100, 5, -1000)

    # how many best optimals to be recorded
    result_size = 5

    # maximum number of iterations of the solver
    max_iter = 500

    # stop the solver when that many times
    # the best optimal have not been changed.
    early_stopping = 100

    # if the model is called from the terminal,
    # for example:  !python file.py
    # this variable should be set as True
    # to display info correctly in Colab
    from_main = False

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print(f"{a:20} {getattr(self, a)}")
        print("\n")
