import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import os

from strategy import run_rotational_strategy
from laddering import compute_laddered_returns, infer_n_ladders

# Define Parameters
Aperiods_stocks = 40  # Short term lookback window)
freq_stocks = "2W-FRI"  # How often the we should rebuild portfolio
delay_stocks = 1  # Days delayed to holdings before returns are taken

Aperiods_bonds = 80  # Short term lookback window)
freq_bonds = "4W-FRI"  # How often the we should rebuild portfolio
delay_bonds = 1  # Days delayed to holdings before returns are taken

# Weights to calculate combined score (MUST sum to 1)
shortTermWeight = 0.4
longTermWeight = 0.3
ShortTermVolatilityWeight = 0.3

# Prefer higher values when scoring? (1= yes, 0 = No)
momentum = 1
volmomentum = 0

# Ticker of cash substitute holding
cashSubstitute = "SHY"

# Laddering Params
LADDERING_ON = 1  # Toggle to turn it on (1) or off (0)
LADDER_STEP = "W-FRI"  # How often should we rebalane a ladder
N_LADDERS = None  # How many ladders (If None, we calc it)

# Paramter to create weighted portforlio (I = stock_weight * I_stock + (1-stock_weight)* I_Bonds)
stock_weight = 0.7


# Path to price data
dfP_stocks = pd.read_csv('Data/UUP.FDN.IBB.IEZ.IGV.IHE.IHF.IHI.ITA.ITB.IYJ.IYT.IYW.IYZ.KBE.KCE.KIE.PBJ.PBS.SMH.VNQ.SHY.csv', parse_dates=['Date'])
dfAP_stocks = pd.read_csv('Data/UUP.FDN.IBB.IEZ.IGV.IHE.IHF.IHI.ITA.ITB.IYJ.IYT.IYW.IYZ.KBE.KCE.KIE.PBJ.PBS.SMH.VNQ.SHY.AP.csv', parse_dates=['Date'])

dfP_bonds = pd.read_csv('Data/BIL.TIP.IEI.IEF.TLH.TLT.SHY.csv', parse_dates=['Date'])
dfAP_bonds = pd.read_csv('Data/BIL.TIP.IEI.IEF.TLH.TLT.SHY.AP.csv', parse_dates=['Date'])

# Sort data by dates
dfP_stocks = dfP_stocks.sort_values('Date').set_index('Date').ffill()
dfAP_stocks = dfAP_stocks.sort_values('Date').set_index('Date').ffill()

dfP_bonds = dfP_bonds.sort_values('Date').set_index('Date').ffill()
dfAP_bonds = dfAP_bonds.sort_values('Date').set_index('Date').ffill()


# Run baseline for stocks and bonds
dfChoice_stocks, dfPRR_stocks = run_rotational_strategy(
    dfP_stocks, dfAP_stocks, Aperiods_stocks,
    shortTermWeight, longTermWeight, ShortTermVolatilityWeight,
    freq_stocks, delay_stocks, momentum, volmomentum, cashSubstitute
)

dfChoice_bonds, dfPRR_bonds = run_rotational_strategy(
    dfP_bonds, dfAP_bonds, Aperiods_bonds,
    shortTermWeight, longTermWeight, ShortTermVolatilityWeight,
    freq_bonds, delay_bonds, momentum, volmomentum, cashSubstitute
)

# Apply laddering
if LADDERING_ON:

    # Determnine number of ladders
    if N_LADDERS is None:
        N_LADDERS_stocks = infer_n_ladders(freq_stocks, LADDER_STEP)
        N_LADDERS_bonds = infer_n_ladders(freq_bonds, LADDER_STEP)
    else:
        N_LADDERS_stocks = N_LADDERS
        N_LADDERS_bonds = N_LADDERS

    # Compute daily laddered portfolio reutrns
    ladder_r_stocks = compute_laddered_returns(
        dfChoice_stocks,
        dfAP_stocks,
        LADDER_STEP,
        N_LADDERS_stocks,
        delay_stocks,
        cashSubstitute,
    )

    ladder_r_bonds = compute_laddered_returns(
        dfChoice_bonds,
        dfAP_bonds,
        LADDER_STEP,
        N_LADDERS_bonds,
        delay_bonds,
        cashSubstitute,
    )

    # Save Results
    dfPRR_stocks["LADDER_ALL_R"] = ladder_r_stocks
    dfPRR_stocks["LADDER_I"] = (1 + ladder_r_stocks.fillna(0.0)).cumprod()
    dfPRR_stocks.loc[dfPRR_stocks.index[0], "LADDER_I"] = 1

    dfPRR_bonds["LADDER_ALL_R"] = ladder_r_bonds
    dfPRR_bonds["LADDER_I"] = (1 + ladder_r_bonds.fillna(0.0)).cumprod()
    dfPRR_bonds.loc[dfPRR_bonds.index[0], "LADDER_I"] = 1


# Combine into single equity curve
common_idx = dfPRR_stocks.index.intersection(dfPRR_bonds.index)

if LADDERING_ON:
    stocks_r = dfPRR_stocks.loc[common_idx, "LADDER_ALL_R"].fillna(0.0)
    bonds_r = dfPRR_bonds.loc[common_idx, "LADDER_ALL_R"].fillna(0.0)
    stock_I = dfPRR_stocks.loc[common_idx, "LADDER_I"].copy()
    bond_I = dfPRR_bonds.loc[common_idx, "LADDER_I"].copy()
else:
    stocks_r = dfPRR_stocks.loc[common_idx, "ALL_R"].fillna(0.0)
    bonds_r = dfPRR_bonds.loc[common_idx, "ALL_R"].fillna(0.0)
    stock_I = dfPRR_stocks.loc[common_idx, "I"].copy()
    bond_I = dfPRR_bonds.loc[common_idx, "I"].copy()

combined_r = stock_weight * stocks_r + (1 - stock_weight) * bonds_r
combined_I = (1 + combined_r).cumprod()
combined_I.loc[combined_I.index[0]] = 1


# Build single output dataframe (dates + all closing prices + equity curves)
# Handle overlapping ticker names by suffixing stock/bond columns
stocks_prices = dfP_stocks.copy()
bonds_prices = dfP_bonds.copy()

overlap = set(stocks_prices.columns).intersection(set(bonds_prices.columns))
if len(overlap) > 0:
    stocks_prices = stocks_prices.rename(columns={c: f"{c}_STK" for c in overlap})
    bonds_prices = bonds_prices.rename(columns={c: f"{c}_BND" for c in overlap})

prices_all = pd.concat([stocks_prices, bonds_prices], axis=1)

# Align prices to the common index used by equity curves
prices_all = prices_all.reindex(common_idx).ffill()

dfOut = pd.DataFrame(index=common_idx)
dfOut = pd.concat([dfOut, prices_all], axis=1)

dfOut["Stock_I"] = stock_I.values
dfOut["Bond_I"] = bond_I.values
dfOut["Combined_I"] = combined_I.values


#  Plot equity curves
plt.figure()
dfOut["Stock_I"].plot(label="Stock_I")
dfOut["Bond_I"].plot(label="Bond_I")
dfOut["Combined_I"].plot(label="Combined_I")
plt.legend()
plt.show()

# Helper fucntions to calculate statistics
def annualized_return(I):
    total_return = I.iloc[-1] / I.iloc[0] - 1
    years = (I.index[-1] - I.index[0]).days / 365.25
    return (1 + total_return) ** (1 / years) - 1

def annualized_vol(r):
    return r.std() * np.sqrt(252)

def sharpe_ratio(r):
    if r.std() == 0:
        return 0.0
    return (r.mean() / r.std()) * np.sqrt(252)

# Compute daily returns from equity curves
stock_r = dfOut["Stock_I"].pct_change().dropna()
bond_r = dfOut["Bond_I"].pct_change().dropna()
combined_r = dfOut["Combined_I"].pct_change().dropna()

print("Stock Final I:   ", round(dfOut["Stock_I"].iloc[-1], 3))
print("Bond Final I:    ", round(dfOut["Bond_I"].iloc[-1], 3))
print("Combined Final I:", round(dfOut["Combined_I"].iloc[-1], 3))
print()

print("CAGR")
print("Stock CAGR:      ", round(annualized_return(dfOut["Stock_I"]) * 100, 2), "%")
print("Bond CAGR:       ", round(annualized_return(dfOut["Bond_I"]) * 100, 2), "%")
print("Combined CAGR:   ", round(annualized_return(dfOut["Combined_I"]) * 100, 2), "%")
print()

print("Volatility (Annualized)")
print("Stock Vol:       ", round(annualized_vol(stock_r) * 100, 2), "%")
print("Bond Vol:        ", round(annualized_vol(bond_r) * 100, 2), "%")
print("Combined Vol:    ", round(annualized_vol(combined_r) * 100, 2), "%")
print()

print("Sharpe Ratio")
print("Stock Sharpe:    ", round(sharpe_ratio(stock_r), 3))
print("Bond Sharpe:     ", round(sharpe_ratio(bond_r), 3))
print("Combined Sharpe: ", round(sharpe_ratio(combined_r), 3))
print()

# Save results to CSV file
os.makedirs("Results", exist_ok=True)
dfOut.to_csv("Results/rotational_multi_portfolio_full.csv", index=True)