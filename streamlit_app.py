import pandas as pd
import numpy as np
import streamlit as st
import yaml
from typing import Union
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title('ESG EzInvest 🌱')


def adjust_risk_for_invest(age: int) -> Union[int, int]:
  """Adjust asset allocation by age.
  https://www.setinvestnow.com/th/knowledge/article/46-porfolio-management-by-age
  Args:
    age (int):
  Return:
    pct_esg_stock (int)
    pct_esg_bond_cash (int):
  """
  if age>=21 and age<=30:
    pct_esg_stock, pct_esg_bond_cash = 90, 10
  elif age>=31 and age<=40:
    pct_esg_stock, pct_esg_bond_cash = 50, 50
  elif age>=41 and age<=55:
    pct_esg_stock, pct_esg_bond_cash = 30, 70
  elif age>55:
    pct_esg_stock, pct_esg_bond_cash = 10, 90
  return pct_esg_stock, pct_esg_bond_cash

# is_robo_invest = st.button("Robo ESG Invest")

# if is_robo_invest:
tickers = [
  'MINT.BK',
  'DELTA.BK',
  'SCC.BK',
  'CIMBT.BK',
  'BANPU.BK',
  'STGT.BK',
  'BTS.BK',
  'PTT.BK',
  'IRPC.BK',
  'BCP.BK'
]
not_div_tickers = ['AAV',
 'ACE',
 'AOT',
 'BWG',
 'CENTEL',
 'CFRESH',
 'ERW',
 'JAS',
 'KEX',
 'PPP',
 'STARK',
 'SUPER',
 'SYNTEC',
 'THAI']

# Initial read pickle file
esg_score_df = pd.read_pickle('esg_score.pkl')
esg_score_df = esg_score_df.sort_values(by='Refinitiv ESG', ascending=False)
all_prices = pd.read_pickle('all_prices.pkl')
market_prices = pd.read_pickle('market_prices.pkl')
financial_info_df = pd.read_pickle('financial_info.pkl')
esg_bond_df = pd.read_pickle('esg_bond.pkl')  # ESG bonds


# Edit to new library to financial number.
# mcaps = {}
# betas = {'BANPU.BK': 1.00,
# 'BCP.BK': 1.14,
# 'BTS.BK': 0.52,
# 'CIMBT.BK': 0.40,
# 'DELTA.BK': 2.06,
# 'IRPC.BK': 1.02,
# 'MINT.BK': 1.13,
# 'PTT.BK': 0.78,
# 'SCC.BK': 0.44,
# 'STGT.BK': 0.20}
# payouts = {'BANPU.BK': 238.33/100,
# 'BCP.BK': 14.51/100,
# 'BTS.BK': 794.87/100,
# 'CIMBT.BK': 19.17/100,
# 'DELTA.BK': 27.78/100,
# 'IRPC.BK': 72.00/100,
# 'MINT.BK': 58.82/100,
# 'PTT.BK': 43.86/100,
# 'SCC.BK': 19.85/100,
# 'STGT.BK': 416.67/100}
# for t in tickers:
#   stock = yf.Ticker(t)
#   mcaps[t] = stock.fast_info["marketCap"]


# Left
col1, col2 = st.columns(2)
with col1:
  initial_capital = st.number_input('Initial Capital:', min_value=3000, max_value=500000)
with col2:
  esg_stock_options = st.multiselect(
    '**Customize your ESG Stock:**',
    options=list(filter(lambda symbol: symbol not in not_div_tickers, esg_score_df['Symbol'].to_list())),
    default=[ticker.replace('.BK', '') for ticker in tickers],
    # key='assets_options'
  )

# Right
col1, col2 = st.columns(2)
with col1:
  age = st.number_input('Age:', min_value=25, max_value=90)
with col2:
  min_ttm_df = esg_bond_df.loc[esg_bond_df.groupby('Symbol')['TTM(Yrs.)'].idxmin()]
  selected_esg_bond_df = min_ttm_df.sort_values('Refinitiv ESG', ascending=False).reset_index(drop=True).head()
  esg_bond_options = st.multiselect(
    '**Customize your ESG Bond [Preview]:**',
    options=esg_bond_df['ThaiBMA Symbol'].to_list(),
    default=selected_esg_bond_df['ThaiBMA Symbol'].to_list(),
    # key='assets_options'
  )


# Constructing input
esg_stock_options_bk = [s + '.BK' for s in esg_stock_options]
_financial_info = financial_info_df[financial_info_df['Symbol'].isin(esg_stock_options_bk)]
prices = all_prices[esg_stock_options_bk]
mcaps = _financial_info.set_index('Symbol')['Marketcap'].to_dict()
betas = _financial_info['Beta (5Y Monthly)']
payouts = _financial_info['Payout Ratio 4']


# Constructing the prior
from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel, plotting
import pypfopt

S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
delta = black_litterman.market_implied_risk_aversion(market_prices)
market_prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)


def spice_model_mapper(esg_score: float, beta: float) -> float:
  """Calulate Views (Adjust Beta using ESG Score with SPICE Model)
  Args:
    esg_score (float):
    beta (float):
  Return:
    beta_adjustment (float):
  """
  if esg_score>=85:                     # -20%
    adjust_beta = beta - (beta*0.2)
  elif esg_score>=80 and esg_score<85:  # -10%
    adjust_beta = beta - (beta*0.1)
  elif esg_score>=75 and esg_score<80:  # 0%
    adjust_beta = beta
  elif esg_score>=70 and esg_score<75:  # +10%
    adjust_beta = beta + (beta*0.1)
  elif esg_score<70:  # +20%
    adjust_beta = beta + (beta*0.2)
  return adjust_beta

selected_column_list = ['Symbol', 'Refinitiv ESG']
# spice_model_df = esg_score_df.sort_values(by='Refinitiv ESG', ascending=False)[selected_column_list].head(10).reset_index(drop=True)
spice_model_df = esg_score_df[esg_score_df['Symbol'].isin([e.replace('.BK', '') for e in esg_stock_options_bk])][['Symbol', 'Refinitiv ESG']].reset_index(drop=True)
spice_model_df['Symbol'] = spice_model_df['Symbol'].apply(lambda row: row+'.BK')
# spice_model_df['Beta'] = pd.Series(betas).to_list()
spice_model_df['Beta'] = betas.to_list()  # Revised version
spice_model_df['Adjusted Beta'] = spice_model_df.apply(lambda row: spice_model_mapper(row['Refinitiv ESG'], row['Beta']), axis=1)

# SET Performance
market_price_1 = market_prices.head(1)[0]   # 2022-10-26
market_price_2 = market_prices.tail(1)[0]   # 2023-10-26
market_performance = (market_price_2 - market_price_1)/market_price_1
spice_model_df['Market Return 1YR'] = market_performance # P.Job request
# Test Convert market_risk_premium=-14%

# Construct views (base version)
spice_model_df['New Expected Return'] = spice_model_df['Adjusted Beta'] * spice_model_df['Market Return 1YR']

# Construct views (new version)
# expected_return = risk_free_rate + (beta * market_risk_premium)
risk_free_rate = 0.012
# market_risk_premium = 0.08-risk_free_rate
market_risk_premium = market_performance-risk_free_rate

# Implied PER (Payout Ratio)/(Ke-g) as Valuation Model
# spice_model_df['payoutRatio'] = pd.Series(payouts).to_list()
spice_model_df['payoutRatio'] = payouts.to_list()  # Revised version
spice_model_df['Base PER'] = spice_model_df['payoutRatio']/((risk_free_rate + (spice_model_df['Beta'] * market_risk_premium))-0.03)
spice_model_df['New PER'] = spice_model_df['payoutRatio']/((risk_free_rate + (spice_model_df['Adjusted Beta'] * market_risk_premium))-0.03)

# Calculate premium as view
spice_model_df['premium'] = (spice_model_df['New PER']-spice_model_df['Base PER'])/spice_model_df['Base PER']
spice_model_df["premium"] = spice_model_df["premium"].apply(lambda row: abs(row))

viewdict = {row['Symbol']: row['premium'] for row in spice_model_df[['Symbol', 'premium']].to_dict('records')}

# We are using the shortcut to automatically compute market-implied prior
bl = BlackLittermanModel(S, pi="market", market_caps=mcaps, absolute_views=viewdict)

ret_bl = bl.bl_returns()
rets_df = pd.DataFrame([market_prior, ret_bl, pd.Series(viewdict)],
            index=["Prior", "Posterior", "Views"]).T
# bl.bl_weights()
S_bl = bl.bl_cov()

from pypfopt import EfficientFrontier, objective_functions

ef = EfficientFrontier(ret_bl, S_bl)
ef.add_objective(objective_functions.L2_reg)
ef.max_sharpe()
weights = ef.clean_weights()
portfolio_performance = ef.portfolio_performance(verbose=False);
# portfolio_performance = ef.portfolio_performance(verbose=True);


# Display portfolio performance
# portfolio_performance = ef.portfolio_performance(verbose=False)
# st.title("Portfolio Performance")
# st.write("Expected Annual Return:", portfolio_performance[0])
# st.write("Annual Volatility:", portfolio_performance[1])
# st.write("Sharpe Ratio:", portfolio_performance[2])

col1, col2, col3 = st.columns(3)
with col1:
  # Display ESG Porfolio pie chart
  
  # col1, col2, col3 = st.columns(3)
  # col1.metric("Stock", "9000", "90%", delta_color="off")
  # col2.metric("Bond", "500", "5%", delta_color="off")
  # col3.metric("Cash", "500", "5%", delta_color="off")

  # fig, ax = plt.subplots(figsize=(8, 8))
  # pd.Series(weights).sort_values(ascending=False).plot.pie(ax=ax)
  # st.pyplot(fig)

  # Adjust weight to add ESG bonds + Cash
  # esg_bond_df = pd.read_pickle('esg_bond.pkl')
  # min_ttm_df = esg_bond_df.loc[esg_bond_df.groupby('Symbol')['TTM(Yrs.)'].idxmin()]
  # selected_esg_bond_df = min_ttm_df.sort_values('Refinitiv ESG', ascending=False).reset_index(drop=True).head()


  pct_esg_stock, pct_esg_bond_cash = adjust_risk_for_invest(age)

  st.subheader("ESG Asset Allocation")
  st.write(f"stock: {pct_esg_stock}%, bond&cash: {pct_esg_bond_cash}%")

  post_weight = {symbol.replace('.BK', ''): weight * (pct_esg_stock/100.0) for symbol, weight in weights.items()}
  stock_post_weight = {symbol: weight * (pct_esg_stock/100.0) for symbol, weight in weights.items()}
  post_weight['Cash'] = (pct_esg_bond_cash/100.0)/2.0
  equal_weight_top5_bonds = (pct_esg_bond_cash/100.0)/2.0
  for bond in selected_esg_bond_df['ThaiBMA Symbol'].to_list():
    post_weight[bond] = equal_weight_top5_bonds/5.0

  fig, ax = plt.subplots(figsize=(8, 8))
  sorted_weights = pd.Series(post_weight).sort_values(ascending=False)
  # Plot a pie chart
  ax.pie(sorted_weights, labels=sorted_weights.index, startangle=0, textprops={'fontsize': 9}, rotatelabels=True)
  # Draw a white circle at the center to create a donut chart
  centre_circle = plt.Circle((0,0),0.65,fc='white')
  fig = plt.gcf()
  fig.gca().add_artist(centre_circle)
  # Add text in the center
  center_text = ax.text(0, 0, 'Asset Allocation', ha='center', va='center', fontsize=10, color='black')
  # Equal aspect ratio ensures that the pie is drawn as a circle
  ax.axis('equal')
  st.pyplot(fig)


# Display ESG Performance (1YR)
with col2:
  st.subheader("ESG Performance (1YR)")
  #===== ESG Performance (1YR) =====#
  from pypfopt import DiscreteAllocation
  da = DiscreteAllocation(stock_post_weight, prices.iloc[0], total_portfolio_value=initial_capital * (pct_esg_stock/100.0))  # Initial capital
  alloc, leftover = da.lp_portfolio(verbose=False)
  # alloc, leftover = da.lp_portfolio(verbose=True)
  # da = DiscreteAllocation(weights, prices.iloc[0], total_portfolio_value=100000)  # Initial capital
  # alloc, leftover = da.lp_portfolio(verbose=True)
  print(f"Leftover: ${leftover:.2f}")

  def alloc_to_value(alloc, prices_for_date):
    asset_values = []

    for asset, allocation in alloc.items():
      if allocation > 0:
        asset_price = prices_for_date[asset]
        asset_value = allocation * asset_price
        asset_values.append(asset_value)

    return asset_values

  import matplotlib.pyplot as plt
  import matplotlib.ticker as mtick
  import numpy as np

  # Add Cash + Top5 ESG bonds to performance 1YR (return yield at redeem/withdraw)
  initial_cap = initial_capital * (pct_esg_bond_cash/100.0)
  ret_bond = (initial_cap/2) * (selected_esg_bond_df['Coupon (%)'].mean()/100.0)
  ret_cash = (initial_cap/2) * (0.015)
  ret_bond_cash = ret_bond + ret_cash
  print('return yield', ret_bond_cash)


  # Calculate portfolio value over time
  portfolio_value = np.zeros(len(prices))
  for i, date in enumerate(prices.index):
    asset_values = alloc_to_value(alloc, prices.loc[date])
    portfolio_value[i] = sum(asset_values) + leftover   # esg stocks
    portfolio_value[i]+= initial_cap                    # Add esg bond + esg cash

  # Add esg bond + esg cash (return yield at redeem/withdraw)
  portfolio_value[i]+= ret_bond_cash


  # Calculate daily returns
  daily_returns = portfolio_value / portfolio_value[0] - 1

  # Calculate daily returns for the benchmark (SET.BK)
  benchmark_daily_returns = market_prices / market_prices.iloc[0] - 1


  # Create a DataFrame with portfolio and benchmark performance
  performance_data = pd.DataFrame({
    'Date': prices.index,
    'Portfolio Returns': daily_returns * 100,
    'Benchmark Returns': benchmark_daily_returns * 100
  })
  performance_data.set_index('Date', inplace=True)


  # Plot portfolio performance
  plt.figure(figsize=(12, 6))
  plt.plot(performance_data.index, performance_data['Portfolio Returns'], label=f"ESG Portfolio: {performance_data['Portfolio Returns'][-1]:.2f}%", color='green', linewidth=.5)
  plt.plot(performance_data.index, performance_data['Benchmark Returns'], label=f"SET: {performance_data['Benchmark Returns'][-1]:.2f}%", color='orange', linewidth=.5)
  plt.xlabel('Date')
  plt.ylabel('Cumulative Returns')
  plt.title('ESG Portfolio vs. SET Performance')
  plt.grid(True)
  plt.legend(loc='upper left')

  # Format y-axis labels as percentages
  plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
  st.pyplot(plt)

with col3:
  # Display ESG Portfolio Breakdown
  st.subheader("ESG Portfolio Breakdown")
  display_weight = {symbol.replace('.BK', ''): (weight * (pct_esg_stock/100.0)) * 100. for symbol, weight in weights.items()}
  stock_weight_df = pd.DataFrame(display_weight.items(), columns=['Symbol', 'Weight (%)']).sort_values(by='Weight (%)', ascending=False).reset_index(drop=True)
  stock_weight_df = stock_weight_df.merge(esg_score_df[['Symbol', 'Refinitiv ESG', 'SET ESG Ratings']], on='Symbol', how='left')
  stock_weight_df = stock_weight_df.rename(columns={'Refinitiv ESG': 'ESG Score'})
  st.dataframe(stock_weight_df[['Symbol', 'ESG Score', 'SET ESG Ratings', 'Weight (%)']], hide_index=True, use_container_width=True)
