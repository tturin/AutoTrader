from prediction_functions import CalculateGBM as GBM
from prediction_functions import CalculateExtrema as Extrema
from trade_functions import DetermineOrderParameters as DOP
from qtpylib.algo import Algo

import sys
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class GBMAlgo(Algo):
    def on_start(self):
        # Direction:
        #   0 = neutral
        #   -1 = bear
        #   1 = bull
        self.direction = 0

        self.current_support = None
        self.current_resistance = None
        self.next_support = None
        self.next_resistance = None

        self.take_profit_level = None
        self.stop_loss_level = None

        # print("ON START BARS:")
        # print(self.bars)
        # print("END START")

    # def on_tick(self, instrument):
    #     print(instrument.get_ticks().columns.values)

    def on_bar(self, instrument):
        bars = instrument.get_bars()
        # print(list(bars))
        # print(bars)
        # print("LEN BARS: ", len(bars))
        if len(bars) < 1000:
            return

        # print("HAVE 20 BARS")

        # print(bars.columns)

        positions = instrument.get_positions()

        # Calculate GBM
        pred_date = bars.index[-1] + dt.timedelta(hours=1)
        print("Prediction date: ", pred_date)
        predictions = GBM.calc_GBM(bars, pred_date, simulations=500)
        mean_pred = predictions['mean'].iloc[0]
        print("Mean prediction: ", mean_pred)
        # mode_pred = predictions.mode(axis=1, numeric_only=None).values[0][0]
        mode_pred = predictions.round(3).mode(axis='columns', dropna=True, numeric_only=True).dropna()
        mode_pred = mode_pred[0].mode()[0]
        print("Mode prediction: ", mode_pred)
        # print("Frequencies: ", predictions.count(axis='columns'))

        # Get new direction
        curr_price = bars['close'].iloc[-1]
        print("Current price: ", curr_price)
        new_direction = None
        if mode_pred > curr_price:
            new_direction = 1
        elif mode_pred == curr_price:
            new_direction = 0
        else:
            new_direction = -1

        print("Previous direction: ", self.direction)
        print("New direction: ", new_direction)

        self.direction = new_direction

        # Calculate supports and resistances
        np_prices = bars['close'].to_numpy()
        date_list = bars.index.to_numpy()
        support_indexes, resistance_indexes = Extrema.calc_support_resistance(np_prices)
        supports = np.take(np_prices, support_indexes)
        resistances = np.take(np_prices, resistance_indexes)

        # Get resistances above current price
        # print("RESISTANCES: ", resistances)
        resistances = np.take(resistances, np.argwhere(resistances > curr_price))
        resistances = np.sort(resistances)
        resistances = np.unique(resistances)
        # print("Above current price: ", resistances)

        # Get supports below current price
        # print("SUPPORTS: ", supports)
        supports = np.take(supports, np.argwhere(supports < curr_price))
        supports = np.sort(supports)
        supports = supports[::-1]
        supports = np.unique(supports)
        # print("Under current price: ", supports)

        print("RESISTANCES: ", resistances)
        print("SUPPORTS: ", supports)

        # Plot stuff
        fig, ax = plt.subplots()
        ax.plot(date_list, np_prices)
        ax.hlines(supports, min(date_list), max(date_list), color='g')
        ax.hlines(resistances, min(date_list), max(date_list), color='r')
        # plt.show()

        # Calculate fibonacci levels
        fib_levels = Extrema.calc_fib_levels(np_prices)
        price_max = np.max(np_prices)
        price_min = np.min(np_prices)
        ax.axhspan(fib_levels[0], price_min, alpha=0.4, color='lightsalmon')
        ax.axhspan(fib_levels[1], fib_levels[0], alpha=0.5, color='palegoldenrod')
        ax.axhspan(fib_levels[2], fib_levels[1], alpha=0.5, color='palegreen')
        ax.axhspan(price_max, fib_levels[2], alpha=0.5, color='powderblue')

        # plt.show()

        # Get closest support or resistance, depending if bull or bear
        # First make sure that there is no current position
        print("POSITIONS: ", positions)

        # Set the 2 closest spreads and resistances
        if supports.size < 2 or resistances.size < 2 or len(fib_levels) == 0:
            print("Not enough supports/resistances/fib levels")
            return

        # Get position type (sell/buy) and modify if needed
        if positions['position'] != 0:
            print("THERE EXISTS A POSITION")
            print(instrument.get_orders())

            if positions['position'] == -1:
                if curr_price < self.current_support:
                    print("Modifying stop loss to current support: ", self.current_support)
                    stop_order = instrument.get_active_order(order_type="STOP")
                    instrument.move_stoploss(self.current_support)

                    print("Modifying take profit to next support: ", self.next_support)
                    # Need to get take profit order ID by filtering out the other orders
                    print(instrument.get_active_order(order_type="MARKET"))

                    # Change current and next support/resistance levels

            elif positions['position'] == 1:
                if curr_price > self.current_resistance:
                    print("Modifying stop loss to current resistance: ", self.current_resistance)
                    stop_order = instrument.get_active_order(order_type="STOP")
                    instrument.move_stoploss(self.current_resistance)
                    print("Modifying take profit to next resistance: ", self.next_resistance)
                    print(instrument.get_active_order(order_type="MARKET"))

            return

        # Decide what order to make and at what levels
        order_params = DOP.levels_order(self.direction, supports, resistances, fib_levels, bars)
        print("ORDER PARAMS: ", order_params)

        if self.direction == -1:
            instrument.sell(quantity=1, target=order_params[1], initial_stop=order_params[0])
        elif self.direction == 1:
            instrument.buy(quantity=1, target=order_params[1], initial_stop=order_params[0])

        # Set current supports/resistances to 1st levels after 2x commission
        commission = order_params[2] / 3
        commission = commission * 2
        support_level = supports[supports < curr_price - commission]
        resistance_level = resistances[resistances > curr_price + commission]

        if support_level.size > 0:
            self.current_support = supports[-1]
        if resistance_level.size > 0:
            self.current_resistance = resistances[0]

        # Set next support/resistance levels as (if exists):
        #   2nd levels after current price + returned commission
        # If does not exist, then do not change take profit levels
        support_level = supports[supports < curr_price - order_params[2]]
        resistance_level = resistances[resistances > curr_price + order_params[2]]

        if support_level.size > 1:
            self.next_support = support_level[-2]
        if resistance_level.size > 0:
            self.next_resistance = resistance_level[1]


if __name__ == "__main__":
    strategy = GBMAlgo(
        instruments=[("AMD", "STK", "SMART", "USD", "", 0.0, "")],
        resolution="15T",
        bar_window=1000,
        # tick_window=1000,
        # preload="1000T",
        preload="10D",
        # timezone='UTC',
        ibport=7497,
        backtest=False,
        start='2020-04-22',
        end='2020-04-23',
        output='C:\\Users\\ogaboga\\PycharmProjects\\AutoTrader\\main\\portfolio.csv'
    )

    strategy.run()
