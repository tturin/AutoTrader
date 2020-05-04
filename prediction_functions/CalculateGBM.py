import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters


def calc_GBM(input_df: pd.DataFrame, pred_end_date, simulations=6):
    # Reset index
    input_df = input_df.reset_index()
    input_df = input_df.rename(columns={"index": "datetime"})
    # print(input_df)

    # Start date and End date
    start_date = input_df['datetime'].iloc[0]
    # start_date = input_df.first_valid_index()
    end_date = input_df['datetime'].iloc[-1]
    # end_date = input_df.last_valid_index()
    # Initial price to multiply by
    # initial_price = input_df['last'].iloc[-1]
    initial_price = input_df['close'].iloc[-1]
    # Time increment
    # dt = 1
    dt = 1
    # Prediction time horizon
    # num_weekdays = pd.date_range(
    #     # start=pd.to_datetime(end_date, format='%Y-%m-%d') + pd.Timedelta('1 days'),
    #     # end=pd.to_datetime(pred_end_date, format='%Y-%m-%d')
    #     start=pd.to_datetime(end_date, format='%Y-%m-%d %H:%M:%S') + pd.Timedelta('1 minutes'),
    #     end=pd.to_datetime(pred_end_date, format='%Y-%m-%d %H:%M:%S')
    # ).to_series().map(lambda x: 1 if x.isoweekday() in range(1, 6) else 0).sum()
    T = (pred_end_date - end_date).total_seconds() / 60.0
    # Number of time points in T
    N = T / dt
    # Array of time progression
    t = np.arange(1, int(N) + 1)
    # Mean return of stock prices in historical data range
    returns = (input_df.loc[1:, 'close'] -
               input_df.shift(1).loc[1:, 'close']) / \
        input_df.shift(1).loc[1:, 'close']
    mu = np.mean(returns)
    # Standard deviation of returns
    sigma = np.std(returns)
    # Randomness array
    scen_size = simulations
    b = {str(scen): np.random.normal(0, 1, int(N)) for scen in range(1, scen_size + 1)}
    # Brownian path
    W = {str(scen): b[str(scen)].cumsum() for scen in range(1, scen_size + 1)}

    # Calculate drift
    drift = (mu - 0.5 * sigma**2) * t
    # Calculate diffusion
    diffusion = {str(scen): sigma * W[str(scen)] for scen in range(1, scen_size + 1)}

    # Calculate predictions
    S = np.array([initial_price * np.exp(drift + diffusion[str(scen)]) for scen in range(1, scen_size + 1)])
    S = np.hstack((np.array([[initial_price] for scen in range(scen_size)]), S))

    # Plot simulations
    # plt.figure(figsize=(20, 10))
    # for i in range(scen_size):
    #     plt.title('Daily Volatility: ' + str(sigma))
    #     # plt.plot(pd.date_range(start=input_df['datetime'].max(),
    #     #                        end=pred_end_date,
    #     #                        freq=None
    #     #                        # ).map(lambda x: x if x.isoweekday() in range(1, 6) else np.nan).dropna(),
    #     #                        ).map(lambda x: x),
    #     #          S[i, :])
    #     plt.plot(pd.date_range(
    #         start=input_df['datetime'].max(),
    #         end=pred_end_date,
    #         freq='T'
    #     ), S[i, :])
    #     plt.ylabel('Stock Prices, $')
    #     plt.xlabel('Prediction Days')


    # plt.show()

    # Convert predictions to DataFrame
    preds_df = pd.DataFrame(S.swapaxes(0, 1)[:, :]).set_index(
        pd.date_range(start=input_df['datetime'].max(),
                    # start=input_df.index.max(),
                      end=pred_end_date,
                      # freq='D').map(lambda x: x if x.isoweekday() in range(1, 6) else np.nan).dropna()
                      freq='T')
    ).reset_index(drop=False)

    preds_df['mean'] = preds_df.mean(axis=1)

    return preds_df
