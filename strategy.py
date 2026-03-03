# strategy.py

import pandas as pd
import numpy as np
import math


def run_rotational_strategy(
    dfP: pd.DataFrame,
    dfAP: pd.DataFrame,
    Aperiods: int,
    ShortTermWeight: float,
    LongTermWeight: float,
    ShortTermVolatilityWeight: float,
    Frequency: str,
    Delay: int,
    momentum: int,
    volmomentum: int,
    StandsForCash: str,
):
    # Define parameters 
    Bperiods = 3 * Aperiods + ((3 * Aperiods) // 20) * 2 # longer lookback winder
    Speriods = Aperiods # Window used for volatility calculations 

    # Compute performance metrics for each asset 
    dfA = dfP.pct_change(periods=Aperiods - 1) # Short term % return
    dfB = dfP.pct_change(periods=Bperiods - 1) # Long term % return 
    dfR = dfP.pct_change() # DAily % returns 
    dfS = dfR.rolling(window=Speriods).std() * math.sqrt(252) # Annulailzed volatility over Speriods 

    # FHelper function to rank within each day
    def rank_rows(df_metric: pd.DataFrame, want_high_best: bool) -> pd.DataFrame:
        out = df_metric.copy()
        out[:] = 0
        for r in range(len(df_metric)):
            arr = df_metric.iloc[r].values
            temp = arr.argsort() if want_high_best else (-arr).argsort()
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(1, len(arr) + 1)
            out.iloc[r] = ranks
        return out

    # Caclcuate ranks for each metric
    dfA_r = rank_rows(dfA, momentum == 1) * ShortTermWeight
    dfB_r = rank_rows(dfB, momentum == 1) * LongTermWeight
    dfS_r = rank_rows(dfS, volmomentum == 1) * ShortTermVolatilityWeight

    # Compute weighted score of ranks
    dfAll = dfA_r + dfB_r + dfS_r

    # Pick best asset each day
    dfChoice = dfAll.copy()
    dfChoice[:] = 0
    for r in range(len(dfAll)):
        max_col = int(np.nanargmax(dfAll.iloc[r].values))
        dfChoice.iat[r, max_col] = 1

    # Compute portoflio returns using adj prices (dfPRR starts as daily return for each asset)
    dfPRR = dfAP.pct_change().copy()

    # Generate rebalance dates and intesect with actual dates in dfP
    rebalance_idx = dfP.asfreq(freq=Frequency, method="pad").index
    rebalance_idx = rebalance_idx[rebalance_idx.isin(dfP.index)]

    dfPRR["REBALANCE"] = False
    dfPRR.loc[rebalance_idx, "REBALANCE"] = True

    # Only hold one asset at a time 
    held = pd.Series(index=dfPRR.index, dtype="object")
    held.iloc[0] = StandsForCash  

    # For each rebalance day
    for d in dfPRR.index[dfPRR["REBALANCE"]]:
        # Select the chosen stock
        selected = dfChoice.loc[d].idxmax()
        held.loc[d] = selected

    # Hold it for the period
    held = held.ffill()

    # Trade delay
    held_exec = held.shift(Delay).ffill().fillna(StandsForCash)

    # Ensure that data is one-hot-encoded (we are only holding one stock at a time)
    for tkr in dfAP.columns:
        dfPRR[tkr + "_NUL"] = (held_exec == tkr).astype(float)

    # Compute returns for each ticker
    for tkr in dfAP.columns:
        dfPRR[tkr + "_R"] = dfPRR[tkr].fillna(0.0) * dfPRR[tkr + "_NUL"]

    # Compute portfolio performance 
    dfPRR["ALL_R"] = dfPRR[[c + "_R" for c in dfAP.columns]].sum(axis=1)
    dfPRR["I"] = (1.0 + dfPRR["ALL_R"]).cumprod()
    dfPRR.loc[dfPRR.index[0], "I"] = 1.0

    return dfChoice, dfPRR