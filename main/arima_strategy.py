from qtpylib.algo import Algo
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from pandas.plotting import lag_plot

from prediction_functions import CalculateExtrema as Extrema
from trade_functions import DetermineOrderParameters as DOP

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import pmdarima as pm


class ARIMAAlgo(Algo):
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

    def on_bar(self, instrument):
        bars_min = instrument.get_bars()
        if len(bars_min) < 1000:
            return

        # Resample data to hourly data
        bars = bars_min.resample('H').mean()
        # plt.figure(figsize=(10, 10))
        # plt.plot(bars_min['close'], label='Hourly Data')
        # plt.plot(bars['close'], label='15 Minute Data')
        # plt.legend()
        # plt.show()

        # Get positions
        positions = instrument.get_positions()

        # Test autocorrelation of close prices
        # plt.figure(figsize=(10, 10))
        # lag_plot(bars['close'], lag=5)
        # plt.title('Autocorrelation Plot')
        # plt.show()

        # Create ARIMA model
        # Split bars into 80% training and 20% testing
        train_data = bars[0:int(len(bars)*0.8)]
        test_data = bars[int(len(bars)*0.8):]
        # plt.figure(figsize=(12,7))
        # plt.title("Stock Close Prices")
        # plt.xlabel('Date')
        # plt.ylabel('Price')
        # plt.plot(bars['close'], 'blue', label='Training Data')
        # plt.plot(test_data['close'], 'green', label='Testing Data')
        # plt.legend()
        # plt.show()

        train_values = train_data['close'].values
        test_values = test_data['close'].values

        # Train data and test data
        history = [x for x in train_values]
        predictions = list()
        for t in range(len(test_values)):
            model = ARIMA(history, order=(5, 1, 0))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            pred = output[0]
            predictions.append(pred)
            obs = test_values[t]
            history.append(obs)

        # ARIMA with auto parameter optimizations
        # for t in range(len(test_values)):
        #     model = pm.auto_arima(history, start_p=1, start_q=1, test='adf',
        #                       d=None, seasonal=False, start_P=0, trace=True,
        #                       error_action='ignore', suppress_warnings=True, stepwise=True)
        #     # print(model.summary())
        #     pred, conf_int = model.predict(n_periods=1, return_conf_int=True)
        #     predictions.append(pred[0])
        #     history.append(pred[0])
        #     print(model.params()[0])
        #     print(model.params()[1])
        #     print(model.params()[2])
        #     return

        mse = mean_squared_error(test_values, predictions)
        print('Testing MSE: %.3f' % mse)
        # smape = smape_calc(test_values, predictions)
        # print('SMAPE: %.3f' % smape)

        # Plot results
        # plt.figure(figsize=(12, 7))
        # plt.plot(bars['close'], 'green', color='blue', label='Training Data')
        # plt.plot(test_data.index, predictions, color='green', marker='o', linestyle='dashed', label='Predicted Price')
        # plt.plot(test_data.index, test_data['close'], color='red', label='Actual Price')
        # plt.title('ARIMA Stock Price Predictions')
        # plt.xlabel('Date')
        # plt.ylabel('Price')
        # plt.legend()
        # plt.show()

        # If Mean Squared Error and SMAPE are within thresholds,
        # execute ARIMA model on all bars
        pred = 0
        if mse < 1:
            model = ARIMA(bars['close'].values, order=(5, 1, 0))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            print("OUTPUT: ", output)
            pred = output[0]
        else:
            return

        curr_price = bars['close'].iloc[-1]

        # Set direction
        new_direction = None
        if pred > curr_price:
            new_direction = 1
        elif pred == curr_price:
            new_direction = 0
        else:
            new_direction = -1
        self.direction = new_direction

        print("Prediction for price in 1 hour: ", pred)
        print("Current price: ", curr_price)
        print("Direction: ", self.direction)

        # Calculate supports and resistances
        np_prices = bars['close'].to_numpy()
        date_list = bars.index.to_numpy()
        support_indexes, resistance_indexes = Extrema.calc_support_resistance(np_prices)
        supports = np.take(np_prices, support_indexes)
        resistances = np.take(np_prices, resistance_indexes)

        # Get resistances above current price
        resistances = np.take(resistances, np.argwhere(resistances > curr_price))
        resistances = np.sort(resistances)
        resistances = np.unique(resistances)

        # Get supports below current price
        supports = np.take(supports, np.argwhere(supports < curr_price))
        supports = np.sort(supports)
        supports = supports[::-1]
        supports = np.unique(supports)

        # Calculate fibonacci levels
        fib_levels = Extrema.calc_fib_levels(np_prices)
        price_max = np.max(np_prices)
        price_min = np.min(np_prices)

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

            # Do not continue since a position exists
            return

        # Determine if the move is large enough
        # If predicted mode price is above/below 2nd resistance/support
        if self.direction == -1:
            if pred > supports[-2]:
                print("Not large enough move")
                print("2nd support: ", supports[-2])
                return
        elif self.direction == 1:
            if pred < resistances[1]:
                print("Not large enough move")
                print("2nd resistance: ", resistances[1])
                return

        # Get order parameters
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
        #   - 2nd levels after current price + returned commission
        # If does not exist, then do not change take profit levels
        support_level = supports[supports < curr_price - order_params[2]]
        resistance_level = resistances[resistances > curr_price + order_params[2]]

        if support_level.size > 1:
            self.next_support = support_level[-2]
        if resistance_level.size > 0:
            self.next_resistance = resistance_level[1]


def smape_calc(y_true, y_pred):
    return np.mean((np.abs(y_pred - y_true) * 200 / (np.abs(y_pred) + np.abs(y_true))))


if __name__ == "__main__":
    strategy = ARIMAAlgo(
        instruments=[("AMD", "STK", "SMART", "USD", "", 0.0, "")],
        resolution="15T",
        bar_window=1000,
        # tick_window=1000,
        # preload="1000T",
        preload="10D",
        # timezone='UTC',
        ibport=7497
        # backtest=False,
        # start='2020-04-22',
        # end='2020-04-23',
        # output='C:\\Users\\ogaboga\\PycharmProjects\\AutoTrader\\main\\portfolio.csv'
    )

    strategy.run()
