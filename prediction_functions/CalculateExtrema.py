from findiff import FinDiff

import numpy as np


def calc_support_resistance(prices):
    dx = 1
    d_dx = FinDiff(0, dx, 1)
    d2_dx2 = FinDiff(0, dx, 2)
    clarr = np.asarray(prices)
    mom = d_dx(clarr)
    momacc = d2_dx2(clarr)

    minimaIdxs, maximaIdxs = get_extrema(prices, True, mom, momacc), get_extrema(prices, False, mom, momacc)
    return minimaIdxs, maximaIdxs


def get_extrema(prices, isMin, mom, momacc):
    return [x for x in range(len(mom))
            if (momacc[x] > 0
                if isMin
                else momacc[x] < 0) and (
                    mom[x] == 0 or (
                    (x != len(mom) - 1) and (
                    mom[x] > 0 and mom[x + 1] < 0 and
                    prices[x] >= prices[x + 1]
                    or mom[x] < 0
                    and mom[x + 1] > 0 and prices[x] <= prices[x + 1])
                    or x != 0 and (mom[x - 1] > 0 and mom[x] < 0 and prices[x - 1] < prices[x]
                    or mom[x - 1] < 0 and mom[x] > 0 and prices[x - 1] > prices[x]
            )))]

def calc_fib_levels(prices):
    price_max = np.max(prices)
    price_min = np.min(prices)

    diff = price_max - price_min
    level_1 = price_max - 0.236 * diff
    level_2 = price_max - 0.382 * diff
    level_3 = price_max - 0.618 * diff

    return level_1, level_2, level_3
