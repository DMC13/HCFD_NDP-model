"""
HCFD Night Duty Planning Model
The main NDP model implementation.

Copyright (c) 2021 Tao-Ming Chen.
Licensed under the LGPL-2.1 License (see LICENSE for details)
"""

import os
import copy
from tqdm import tqdm
import numpy as np
import pandas as pd
import pulp
from pulp import LpProblem, lpSum, LpMinimize, LpMaximize, LpVariable, value, LpStatus


#################################
#   輸入格式定義
#################################


class NDPInput(object):
    """
    create a NDPInput object with the pre-defined format:
    Input = NDPInput({your filepath}, {x_table}, {members_info}, {date_info}, {y})

    - num_dayoff:   an integer. 所有隊員、在此月的總放假日數 
    - members_id:   a list/tuple of integers. 隊員的番號
    - n_members:    an integer. 隊員數量
    - days:         a list/tuple of integers. 日期
    - n_days:       an integer. 本月天數
    """

    def __init__(self, filename, x_table, y, members_info, date_info):

        self.filename = filename
        self.x_table = x_table
        self.y = y

        members_id, n_members, squad = members_info
        days, n_days, shift_off = date_info

        self.members_id = members_id
        self.n_members = n_members
        self.squad = squad
        self.days = days
        self.n_days = n_days
        self.shift_off = shift_off

        self.num_dayoff = self.get_num_of_total_dayoff(x_table)

    def get_num_of_total_dayoff(self, arr):
        # 計算所有隊員，在此月總共休了幾天假
        h, w = arr.shape
        return h*(w-1) - arr[:, 1:].sum()


#################################
#   可行解格式定義
#################################


class AnOptimal(object):
    def __init__(self, var_min, var_max, var_mean, D, E):
        self.var_min = var_min
        self.var_max = var_max
        self.var_mean = var_mean
        self.solution = tuple([copy.deepcopy(D), copy.deepcopy(E)])

    def compare(self, bests):
        # links to BestOpitmals.compare_with_bests(AnOptimal)
        compare_result = bests.compare_with_bests(self)
        return compare_result


class BestOptimals(object):
    def __init__(self, size_lim=5, verbose=False):
        self.size_lim = size_lim
        self.last_ind = -1
        self.best_k = list()
        self.verbose = verbose

    def update_bests(self, new_optimal, insert_at):
        """
        insert the new optimal at assigned place.

        new_optimal:    an AnOptimal object
        insert_at:      integer
        """
        if insert_at == self.size_lim:
            return False
        print(
            f'this optimal is the <{insert_at+1}> best.') if self.verbose > 1 else ""
        self.best_k.insert(insert_at, new_optimal)
        self.last_ind += 1
        if self.last_ind == self.size_lim:
            self.best_k.pop()
            self.last_ind -= 1
        return True

    def compare_with_bests(self, new_optimal, compare_with=None):
        """
        a recursive function aims to compare the new_optimal
        with found best optimals (the elements in best_k).

        returns a boolean value that indicates whether 
        the new_optimal is updated to the best_k.

        a new_optimal is first compared with the last (worst) optimal in best_k,
        if the new_optimal is better than the last one, 
        it is compared with the second last one, and so on.

        - compare_with: an integer representing the index of element that is 
                        currently being compared with.

        special thanks to PJ HUANG for providing valuable insights for this function!
        """

        # if enter this function for the first time, compare with the last one
        if compare_with is None:
            compare_with = self.last_ind

        print(f'comparing with <{compare_with}>') if self.verbose > 1 else ""
        # if new_optimal is the best
        if compare_with < 0:
            has_updated = self.update_bests(new_optimal, 0)
        # else, compare with the other best optimals
        else:
            to_compare = self.best_k[compare_with]
            # var_min
            if new_optimal.var_min > to_compare.var_min:
                has_updated = self.compare_with_bests(
                    new_optimal, (compare_with-1))
            elif new_optimal.var_min == to_compare.var_min:
                # var_max
                if new_optimal.var_max < to_compare.var_max:
                    has_updated = self.compare_with_bests(
                        new_optimal, (compare_with-1))
                elif new_optimal.var_max == to_compare.var_max:
                    # var_mean
                    if new_optimal.var_mean < to_compare.var_mean:
                        has_updated = self.compare_with_bests(
                            new_optimal, (compare_with-1))
                    else:
                        has_updated = self.update_bests(
                            new_optimal, compare_with+1)
                else:
                    has_updated = self.update_bests(
                        new_optimal, compare_with+1)
            else:
                has_updated = self.update_bests(new_optimal, compare_with+1)
        return has_updated


#################################
#   Night Duty Planning Model
#################################


class NDPModel(object):
    def __init__(self, input, config):
        """
        - input:    an NDPInput object.
        - config:   a Config object.
        """

        self.input = input
        self.config = config
        self.generate_scores_array(input, config.score)
        self.build(input)

    def generate_scores_array(self, input, score):
        """
        - input:    a NDPInput object.
        - score:    a list/tuple of four numerical elements.
                    四個數字，依序分別代表將夜勤安排於：
                    休假日、第一天班大夜救護、第一天班夜宿、前一天有班之日子，模型所獲得的分數
        """
        scores_D = np.zeros((input.n_members, input.n_days), dtype='int32')
        scores_E = np.zeros((input.n_members, input.n_days), dtype='int32')
        dayoff, rotate_day, first_day, second_day = score

        for row, member_row in enumerate(input.x_table):
            for col, status in enumerate(member_row[1:]):
                # 若這天休假
                if status == 0:
                    scores_D[row, col] = dayoff
                    scores_E[row, col] = dayoff
                # 若今天有服勤、且前一天沒有服勤
                elif member_row[col] == 0:
                    # 若前一天是輪休班
                    if input.squad[row] == input.shift_off[col]:
                        scores_D[row, col] = (first_day + rotate_day) / 2
                        scores_E[row, col] = rotate_day
                    # 若前一天不是輪休班
                    else:
                        scores_D[row, col] = first_day
                        scores_E[row, col] = first_day
                # 若這天有服勤、且前一天有服勤
                else:
                    scores_D[row, col] = second_day
                    scores_E[row, col] = second_day

        self.scores_D, self.scores_E = scores_D, scores_E

    def build(self, input):
        """
        builds a pulp LpProblem model, and create related model variables.

        - input:    a NDPInput object.
        """

        LPmodel = LpProblem("night_duty_problem", LpMaximize)

        #################################
        #   變數
        #################################

        # 變數：值宿
        self.D = LpVariable.dicts("D", ((d, m)
                                        for d in input.days for m in input.members_id), lowBound=0, cat="Binary")
        # 變數：大夜救護
        self.E = LpVariable.dicts("E", ((d, m)
                                        for d in input.days for m in input.members_id), lowBound=0, cat="Binary")

        # Parameters for controlling the behaviour of basic variables
        self.nb = len(input.days)*4 + len(input.members_id)*4 + input.num_dayoff * \
            2 + len(input.days[:-1])*len(input.members_id)
        self.M = 10000

        # Non-negative Slack Variables - one for each constraint
        self.S_daysum_D = LpVariable.dicts(
            's_daysum_d', ((d) for d in input.days), lowBound=0, cat='Continuous')
        self.S_daysum_E = LpVariable.dicts(
            's_daysum_e', ((d) for d in input.days), lowBound=0, cat='Continuous')
        self.S_dayoff = LpVariable.dicts('s_dayoff', ((
            d, m) for d in input.days for m in input.members_id), lowBound=0, cat='Continuous')
        self.S_memsum_D = LpVariable.dicts(
            's_memsum_d', ((m) for m in input.members_id), lowBound=0, cat='Continuous')
        self.S_memsum_E = LpVariable.dicts(
            's_memsum_e', ((m) for m in input.members_id), lowBound=0, cat='Continuous')
        self.S_in_2days = LpVariable.dicts('s_in_2days', ((
            d, m) for d in input.days[:-1] for m in input.members_id), lowBound=0, cat='Continuous')

        # Basis variables (binary)
        # one for each variable & one for each constraint (& so slack)
        self.B_D = LpVariable.dicts(
            'B_d', ((d, m) for d in input.days for m in input.members_id), cat='Binary')
        self.B_E = LpVariable.dicts(
            'B_e', ((d, m) for d in input.days for m in input.members_id), cat='Binary')
        self.B_S_daysum_D = LpVariable.dicts(
            'B_s_daysum_d', ((d) for d in input.days), cat='Binary')
        self.B_S_daysum_E = LpVariable.dicts(
            'B_s_daysum_e', ((d) for d in input.days), cat='Binary')
        self.B_S_dayoff = LpVariable.dicts(
            'B_s_dayoff',   ((d, m) for d in input.days for m in input.members_id), cat='Binary')
        self.B_S_memsum_D = LpVariable.dicts(
            'B_s_memsum_d', ((m) for m in input.members_id), cat='Binary')
        self.B_S_memsum_E = LpVariable.dicts(
            'B_s_memsum_e', ((m) for m in input.members_id), cat='Binary')
        self.B_S_in_2days = LpVariable.dicts(
            'B_s_in_2days', ((d, m) for d in input.days[:-1] for m in input.members_id), cat='Binary')

        #################################
        #   目標式：最大化得分
        #################################

        LPmodel += lpSum(self.scores_D[m-input.members_id[0], d-input.days[0]] * self.D[d, m] + self.scores_E[m -
                                                                                                              input.members_id[0], d-input.days[0]] * self.E[d, m] for d in input.days for m in input.members_id)
        #################################
        #   限制式
        #################################

        # 每日一個值宿
        for d in input.days:
            LPmodel += lpSum(self.D[d, m] for m in input.members_id) + \
                self.S_daysum_D[d] == 1, ("a D a day < "+str(d))
            LPmodel += lpSum(self.D[d, m] for m in input.members_id) - \
                self.S_daysum_D[d] == 1, ("a D a day > "+str(d))

        # 每日兩個大夜救護
        for d in input.days:
            LPmodel += lpSum(self.E[d, m] for m in input.members_id) + \
                self.S_daysum_E[d] == 2, ("two E a day < "+str(d))
            LPmodel += lpSum(self.E[d, m] for m in input.members_id) - \
                self.S_daysum_E[d] == 2, ("two E a day > "+str(d))

        # 休假日不可排夜勤
        for d in input.days:
            for m in input.members_id:
                if input.x_table[m-input.members_id[0], d] == 0:
                    LPmodel += self.D[d, m]+self.E[d, m] + self.S_dayoff[d,
                                                                         m] == 0, ("dayoff < "+str(d)+"-"+str(m))
                    LPmodel += self.D[d, m]+self.E[d, m] - self.S_dayoff[d,
                                                                         m] == 0, ("dayoff > "+str(d)+"-"+str(m))
        # 每人每兩日最多只能一次夜勤
        for d in input.days[:-1]:
            for m in input.members_id:
                LPmodel += self.D[d, m]+self.E[d, m]+self.D[d+1, m]+self.E[d+1, m] + \
                    self.S_in_2days[d, m] == 1, ("in 2days "+str(d)+"-"+str(m))

        # 本月每個人的目標天數
        for ind_m, m in enumerate(input.members_id):
            LPmodel += lpSum(self.D[d, m] for d in input.days) + \
                self.S_memsum_D[m] == input.y[ind_m, 0], ("month D < "+str(m))
            LPmodel += lpSum(self.D[d, m] for d in input.days) - \
                self.S_memsum_D[m] == input.y[ind_m, 0], ("month D > "+str(m))
            LPmodel += lpSum(self.E[d, m] for d in input.days) + \
                self.S_memsum_E[m] == input.y[ind_m, 1], ("month E < "+str(m))
            LPmodel += lpSum(self.E[d, m] for d in input.days) - \
                self.S_memsum_E[m] == input.y[ind_m, 1], ("month E > "+str(m))

        # No. of basics is correct:
        LPmodel += lpSum(self.B_D) + lpSum(self.B_E) + lpSum(self.B_S_daysum_D) + lpSum(self.B_S_daysum_E) + \
            lpSum(self.B_S_dayoff) + lpSum(self.B_S_memsum_D) + \
            lpSum(self.B_S_memsum_E) + lpSum(self.B_S_in_2days) == self.nb
        # Enforce basic and non-basic behaviour
        for i in self.D:
            LPmodel += self.D[i] <= self.M*self.B_D[i]
        for i in self.E:
            LPmodel += self.E[i] <= self.M*self.B_E[i]
        for i in self.S_daysum_D:
            LPmodel += self.S_daysum_D[i] <= self.M*self.B_S_daysum_D[i]
        for i in self.S_daysum_E:
            LPmodel += self.S_daysum_E[i] <= self.M*self.B_S_daysum_E[i]
        for i in self.S_dayoff:
            LPmodel += self.S_dayoff[i] <= self.M*self.B_S_dayoff[i]
        for i in self.S_memsum_D:
            LPmodel += self.S_memsum_D[i] <= self.M*self.B_S_memsum_D[i]
        for i in self.S_memsum_E:
            LPmodel += self.S_memsum_E[i] <= self.M*self.B_S_memsum_E[i]
        for i in self.S_in_2days:
            LPmodel += self.S_in_2days[i] <= self.M*self.B_S_in_2days[i]

        self.LPmodel = LPmodel

    #################################
    #   可行解衡量
    #################################

    def get_dates_of_DE(self, input):
        """
        returns two lists that record the dates of night duty (D) and EMS (E) respectively.
        each list is composed of lists that record the dates of duties of a member.
        """
        dates_of_D = [[] for _ in range(len(input.members_id))]
        dates_of_E = [[] for _ in range(len(input.members_id))]
        for d in input.days:
            for i, m in enumerate(input.members_id):
                if self.D[d, m].varValue == 1:
                    dates_of_D[i].append(d)
                if self.E[d, m].varValue == 1:
                    dates_of_E[i].append(d)
        return dates_of_D, dates_of_E

    def calculate_variance(self, date_D, date_E, weight=(2, 1)):
        """
        given the dates of night duty (D) and EMS (E) of a member,
        calcualtes the variance of D and the variance of E.

        returns the weighted sum of var(D) and var(E).

        若某人被安排：在第3、12天值宿，在第7、16、24、29天大夜救護，
        則 var(D) = var{3, 12} = 40.5； var(E) = var{7, 16, 24, 29} = 92.667
        """
        if len(date_D) < 2 or len(date_E) < 2:
            return None
        else:
            mean_D = sum(date_D)/len(date_D)
            var_D = sum((d-mean_D)**2 for d in date_D)/(len(date_D)-1)

            mean_E = sum(date_E)/len(date_E)
            var_E = sum((e-mean_E)**2 for e in date_E)/(len(date_E)-1)

            return weight[0]*var_D + weight[1]*var_E

    def evaluate_result(self, input, config):
        """
        caculate the variance value of each member,
        than find the min, max, mean of the variance values.
        """
        dates_of_D, dates_of_E = self.get_dates_of_DE(input)
        variance = list()
        for date_D, date_E in zip(dates_of_D, dates_of_E):
            var = self.calculate_variance(
                date_D, date_E, config.variance_weight)
            if var is not None:
                variance.append(var)
        var_min, var_max, var_mean = min(variance), max(
            variance), sum(variance)/len(variance)
        if config.verbose > 1:
            print(
                f'\t-> variance score:\tmin: {var_min:.2f}\tmax: {var_max:.2f}\tmean: {var_mean:.2f}')
        return var_min, var_max, var_mean

    #################################
    #   求解引擎
    #################################

    def solve(self):
        self.bests = self.find_multi_optimals()

    def find_multi_optimals(self):
        model = self.LPmodel
        config = self.config
        bests = BestOptimals(config.result_size, config.verbose)

        found_optimals = list()
        since_last_new_optimal = 0

        print("\nSolving night duty problem...\n") if config.verbose else ""
        with tqdm(total=config.max_iter) as pbar:
            for iter in range(config.max_iter):
                if since_last_new_optimal < config.early_stopping:
                    pbar.update(1)
                    if config.from_main and iter % 20 == 19:
                        # the tqdm bar cannot not be displayed
                        # if this function is called in terminal environment
                        # by a .py file, for example: !python file.py.
                        # so we print another progress bar in a new line.
                        print('')
                    model.solve(pulp.apis.PULP_CBC_CMD(msg=False))
                    print(
                        f'\n#{iter} {pulp.LpStatus[model.status]}\tObjective: {pulp.value(model.objective)}') if config.verbose > 1 else ""

                    if pulp.LpStatus[model.status] == 'Optimal':
                        vars, optimal = list(), list()

                        # parse the optimal value of current iteration
                        for v in model.variables():
                            # get the basis variables of current iteration
                            if v.name.startswith('B') and v.varValue == 1:
                                vars.append(v)
                            # get the duty assignment result of current iteration
                            elif (v.name.startswith('D') or v.name.startswith('E')) and v.varValue == 1:
                                optimal.append(v.name)
                        # add a new constraint to not find this optimal again
                        model += lpSum(v for v in vars) <= self.nb-1

                        if optimal not in found_optimals:
                            # update the found_optimals
                            found_optimals.append(optimal)
                            print(
                                f'\t-> new optimal !\t# {len(found_optimals)}') if config.verbose > 1 else ""

                            # evaluate this optimal, and update early_stopping
                            evaluate_min_var, evaluate_max_var, evaluate_mean = self.evaluate_result(
                                self.input, self.config)
                            print(
                                f'\t-> worst score:\tmin_variance: {evaluate_min_var:.3f}\tmax_variance: {evaluate_max_var:.3f}\tmean of variance: {evaluate_mean:.3f}') if config.verbose > 1 else ""

                            solution = AnOptimal(
                                evaluate_min_var, evaluate_max_var, evaluate_mean, self.D, self.E)
                            compare_result = solution.compare(bests)

                            # if this new optimal is one of the bests,
                            # reset the since_last_new_optimal counter.
                            if compare_result:
                                print(
                                    '\n\t-> bests updated !!!\n') if config.verbose > 1 else ""
                                since_last_new_optimal = 0
                            else:
                                since_last_new_optimal += 1

                        else:
                            since_last_new_optimal += 1

                    # if there is no more optimal solution
                    elif iter == 0:
                        assert AssertionError('\n--> 沒有可行解，請重新編排應值宿、應救護的次數！')
                    else:
                        print(
                            f'\n\nAll optimal(s) were found. End of process.\nSaved best top {bests.last_ind +1} solutions.\n\n stopped at:') if config.verbose else ""
                        return bests
                # if the best optimals haven't been updated
                # for that many consective iterations
                else:
                    stopped_at = '\n\n stopped at:' if config.from_main else ''
                    print(
                        f'\n\nProcess ended with early stopping.\nSaved best top {bests.last_ind +1} solutions.{stopped_at}') if config.verbose else ""
                    return bests
        print(
            f'\n\nProcess ended after {config.max_iter} iterations.\nSaved best top {bests.last_ind +1} solutions.') if config.verbose else ""
        return bests
