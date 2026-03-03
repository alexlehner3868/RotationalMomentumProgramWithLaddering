# laddering.py

import numpy as np
import pandas as pd
import math

# Function to infer the number of ladders based on holding frequency and ladder step
def infer_n_ladders(freq: str, ladder_step: str) -> int:
    # Ensure frequency is weekly based 
    if "W" not in freq:
        raise ValueError("Frequency must be weekly like '2W-FRI'")

    # Extract the number of weeks from holding freq
    left = freq.split("W")[0]
    holding_weeks = 1 if left == "" else int(left)

    # Extract the number of weeks from ladder step
    step_left = ladder_step.split("W")[0]
    step_weeks = 1 if step_left == "" else int(step_left)

    # Number of ladders = holding period / ladder step
    num_ladders = int(math.ceil(holding_weeks / step_weeks))
    return max(1, num_ladders)


# Function to compute daily returns for laddered portfolio
def compute_laddered_returns(
    df_choice: pd.DataFrame,
    df_adj_prices: pd.DataFrame,
    ladder_step_freq: str,
    n_ladders: int,
    delay: int,
    cash_ticker: str,
):
    # Compute the daily asset returns 
    asset_r = df_adj_prices.pct_change().copy()

    # Create the weekly schedule of rebalancing dates 
    ladder_calendar = df_adj_prices.asfreq(freq=ladder_step_freq, method="pad").index # All possible dates
    ladder_calendar = ladder_calendar[ladder_calendar.isin(df_adj_prices.index)] # Need to be in price index

    tickers = list(df_adj_prices.columns)

    # Helper function to pick best stock on the rebalance date
    def pick_ticker(choice_row):
        # No valid choices --> Pick Cash 
        if choice_row.isna().all():
            return cash_ticker
        vals = choice_row.values
        # All are zero --> Pick Cash
        if np.nansum(vals) <= 0:
            return cash_ticker
        # Pick stock with max score 
        return tickers[int(np.nanargmax(vals))]

    # Save the returns for each ladder
    ladder_returns = []

    # Build each ladder independently 
    for k in range(n_ladders):
        # Stagger the rebalancing dates (choose every kth date)
        reb_dates = ladder_calendar[k::n_ladders]
        reb_dates = reb_dates[reb_dates.isin(df_choice.index)]

        # Track what this ladder is holding each day
        held = pd.Series(index=df_adj_prices.index, dtype="object")
        held.iloc[0] = cash_ticker

        # Get the ticker we want to invest in on each rebalancing date
        for d in reb_dates:
            held.loc[d] = pick_ticker(df_choice.loc[d])

        # Hold the stock selected for that day for the period
        held = held.ffill()
        held_exec = held.shift(delay)

        # Compute daily returns for this ladder
        tr = pd.Series(0.0, index=df_adj_prices.index)
        for tkr in tickers:
            mask = (held_exec == tkr)
            tr.loc[mask] = asset_r.loc[mask, tkr]

        ladder_returns.append(tr.fillna(0.0))

    # Combine returns of each ladder (equal weighted)
    ladder_all_r = sum(ladder_returns) / float(n_ladders)
    ladder_all_r.name = "LADDER_ALL_R"

    return ladder_all_r