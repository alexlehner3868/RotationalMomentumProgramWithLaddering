# main.py

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import os

from strategy import run_rotational_strategy
from laddering import compute_laddered_returns, infer_n_ladders

# Define Parameters
Aperiods = 40 # Short term lookback window)
Frequency = "2W-FRI" # How often the we should rebuild portfolio
Delay = 1 # Days delayed to holdings before returns are taken

# Weights to calculate combined score (MUST sum to 1)
ShortTermWeight = 0.4
LongTermWeight = 0.3
ShortTermVolatilityWeight = 0.3

# Prefer higher values when scoring? (1= yes, 0 = No)
momentum = 1
volmomentum = 0

# Ticker of cash substitute holding 
StandsForCash = "SHY"

# Laddering Params 
LADDERING_ON = 1 # Toggle to turn it on (1) or off (0)
LADDER_STEP = "W-FRI" # How often should we rebalane a ladder
N_LADDERS = None # How many ladders (If None, we calc it)

# Paths to Price Data
dfP = pd.read_csv('Data/UUP.FDN.IBB.IEZ.IGV.IHE.IHF.IHI.ITA.ITB.IYJ.IYT.IYW.IYZ.KBE.KCE.KIE.PBJ.PBS.SMH.VNQ.SHY.csv', parse_dates=['Date'])
dfAP = pd.read_csv('Data/UUP.FDN.IBB.IEZ.IGV.IHE.IHF.IHI.ITA.ITB.IYJ.IYT.IYW.IYZ.KBE.KCE.KIE.PBJ.PBS.SMH.VNQ.SHY.AP.csv', parse_dates=['Date'])

# Sort Data by Date 
dfP = dfP.sort_values('Date').set_index('Date').ffill()
dfAP = dfAP.sort_values('Date').set_index('Date').ffill()

# Run baseline rotational momentum strategy 
dfChoice, dfPRR = run_rotational_strategy(
    dfP,
    dfAP,
    Aperiods,
    ShortTermWeight,
    LongTermWeight,
    ShortTermVolatilityWeight,
    Frequency,
    Delay,
    momentum,
    volmomentum,
    StandsForCash,
)

# Apply laddering 
if LADDERING_ON:

    # Determnine number of ladders 
    if N_LADDERS is None:
        N_LADDERS = infer_n_ladders(Frequency, LADDER_STEP)

    # Compute daily laddered portfolio reutrns 
    ladder_r = compute_laddered_returns(
        dfChoice,
        dfAP,
        LADDER_STEP,
        N_LADDERS,
        Delay,
        StandsForCash,
    )

    # Save Results
    dfPRR["LADDER_ALL_R"] = ladder_r
    dfPRR["LADDER_I"] = (1 + ladder_r).cumprod()
    dfPRR.loc[dfPRR.index[0], "LADDER_I"] = 1

# Helper function to calculate sharpe ratio
def sharpe(r):
    r = r.dropna()
    if r.std() == 0 or np.isnan(r.std()):
        return 0.0
    return (r.mean() / r.std()) * math.sqrt(252)

print("Baseline Sharpe:", sharpe(dfPRR["ALL_R"]))
if LADDERING_ON:
    print("Ladder Sharpe:", sharpe(dfPRR["LADDER_ALL_R"]))

#  Plot equity curves
plt.figure()
dfPRR["I"].plot(label="Baseline")
if LADDERING_ON:
    dfPRR["LADDER_I"].plot(label="Laddered")

plt.legend()
plt.show()

# Save results to CSV file
os.makedirs("Results", exist_ok=True)
dfPRR.to_csv("Results/rotational_momentum_with_laddering.csv",
             index=True)