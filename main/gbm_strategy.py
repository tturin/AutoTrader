from prediction_functions import CalculateGBM as GBM
from prediction_functions import CalculateExtrema as Extrema
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
        print("LEN BARS: ", len(bars))
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

        plt.show()

        # Get closest support or resistance, depending if bull or bear
        # First make sure that there is no current position
        print("POSITIONS: ", positions)

        if positions['position'] == 0 and resistances.size > 0 and supports.size > 0:
            if self.direction == 1:
                curr_resistance_index = np.abs(resistances - curr_price).argmin()
                self.current_resistance = resistances.flat[curr_resistance_index]
                if curr_resistance_index + 1 >= resistances.size:
                    self.next_resistance = self.current_resistance
                else:
                    self.next_resistance = resistances.flat[curr_resistance_index + 1]

                # Set take profit level by getting fibonacci level
                # closest to next resistance but above current resistance
                fib_levels = np.asarray(fib_levels)
                fib_above = np.argwhere(fib_levels > self.current_resistance)
                print("FIB ABOVE: ", fib_above)
                take_profit_index = np.abs(fib_above - self.next_resistance).argmin()
                self.take_profit_level = fib_above.flat[take_profit_index]
                print("Take profit: ", self.take_profit_level)

                # Set stop loss level by getting first fibonacci level
                # below current price
                fib_below = np.argwhere(fib_levels < curr_price)
                print("FIB BELOW: ", fib_below)
                if fib_below.size > 0:
                    stop_loss_index = np.abs(fib_below - curr_price).argmin()
                    self.stop_loss_level = fib_below.flat[stop_loss_index]
                else:
                    self.stop_loss_level = supports[0]
                print("Stop loss: ", self.stop_loss_level)

                # Send order
                instrument.buy(quantity=1, target=self.take_profit_level, initial_stop=self.stop_loss_level)

            elif self.direction == -1:
                abs_calc = np.abs(supports - curr_price)
                curr_support_index = None
                if abs_calc.size:
                    curr_support_index = abs_calc.argmin()

                if curr_support_index is not None:
                    self.current_support = supports.flat[curr_support_index]
                    if curr_support_index + 1 >= supports.size:
                        self.next_support = self.current_support
                    else:
                        self.next_support = supports.flat[curr_support_index + 1]
                    print("CURRENT SUPPORT: ", self.current_support)
                    print("NEXT SUPPORT: ", self.next_support)

                    # Set take profit level by getting next fib level
                    # closest to next support but less than current support
                    fib_levels = np.asarray(fib_levels)
                    fib_below = np.argwhere(fib_levels < self.current_support)
                    fib_below_support = None
                    if self.next_support is not None:
                        fib_below_support = np.abs(fib_below - self.next_support)

                    # If there is no fib level under current support,
                    # use next support
                    if fib_below_support is not None and fib_below_support.size:
                        self.take_profit_level = fib_below_support.argmin()
                    else:
                        self.take_profit_level = self.next_support
                    print("Take profit: ", self.take_profit_level)

                    # Set stop loss level by getting next fib level
                    # above current price
                    fib_above = np.argwhere(fib_levels > curr_price)
                    fib_above_price = np.abs(fib_above - curr_price)

                    # If there is no fib above price, use a resistance level above
                    if fib_above_price.size:
                        # self.stop_loss_level = fib_above_price.argmin()
                        self.stop_loss_level = fib_above_price[len(fib_above_price) - 1][0]
                    else:
                        self.stop_loss_level = resistances.flat[0]
                    # print("FIB ABOVE: ", fib_above)
                    # print("FIB ABOVE PRICE: ", fib_above_price)
                    print("Stop loss: ", self.stop_loss_level)

                    # Send order
                    instrument.sell(quantity=1, target=self.take_profit_level, initial_stop=self.stop_loss_level)


if __name__ == "__main__":
    strategy = GBMAlgo(
        instruments=[("UAA", "STK", "SMART", "USD", "", 0.0, "")],
        resolution="1T",
        bar_window=1000,
        # tick_window=1000,
        # preload="1000T",
        preload="12H",
        # timezone='UTC',
        ibport=7497,
        backtest=False,
        start='2020-04-22',
        end='2020-04-23',
        output='C:\\Users\\ogaboga\\PycharmProjects\\AutoTrader\\main\\portfolio.csv'
    )

    strategy.run()
