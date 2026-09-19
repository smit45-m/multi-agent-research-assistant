# Financial Engineering & Quantitative Analytics: Technical Reference

## High-Frequency Order Book Matching

Exchange matching engines maintain a limit order book per instrument,
matching with price-time priority: best price first, FIFO within a price
level. Latency-critical paths are measured tick-to-trade — market data in
to order out — with leading firms achieving sub-microsecond wire-to-wire
via fpga acceleration of feed parsing, book building, and pre-trade risk.
Software engines use lock-free ring buffers, kernel bypass NICs, cache-line
aligned book structures, and busy-poll threads pinned to isolated cores.
Determinism (bounded jitter) matters as much as mean latency; queue
position modeling drives passive fill probability.

## Options Pricing: Black-Scholes and Beyond

Black-Scholes-Merton assumes constant volatility, yet market prices imply
an implied volatility smile across strikes — evidence of fat tails and
jumps. Local volatility fits the smile exactly via the dupire equation,
extracting sigma(K,T) from the option surface, but misprices forward smile
dynamics; stochastic volatility (Heston, SABR) captures dynamics with
semi-analytic pricing. greeks calculation proceeds analytically where
possible, else by adjoint algorithmic differentiation. Path-dependent and
multi-asset payoffs use monte carlo simulation with variance reduction;
early-exercise and barrier products use a pde solver (Crank-Nicolson,
ADI schemes) on structured grids.

## VaR and Expected Shortfall

Value at Risk states a loss quantile — e.g. one-day VaR at confidence
level 99 — but ignores tail shape beyond the quantile and is not
subadditive. Expected shortfall (conditional var) averages losses beyond
VaR, is coherent, and anchors FRTB capital at ES 97.5. Estimation:
historical simulation replays empirical scenarios (capturing fat tails
without distributional assumptions), parametric methods assume normality
and understate tails, filtered historical simulation adds GARCH scaling.
Basel backtesting counts days losses exceed VaR: each backtesting exception
beyond expectation escalates multipliers; too few exceptions signal
over-conservatism.

## Factor Investing

Quantitative factor portfolios harvest premia: momentum (12-1 month
returns), value, quality, low-volatility. The fama-french framework
(market, size, value, later profitability and investment) is the standard
attribution lens. Skill is judged by the information ratio (active return
over tracking error) versus the sharpe ratio for total risk-adjusted
return. Implementation drag matters: a turnover constraint bounds trading
costs, especially for fast signals like momentum; factor correlation
across styles (value-size overlap, momentum-value negative correlation)
drives multi-factor construction toward integrated optimization rather
than sleeve blending, with crowding monitored via spread and ownership
metrics.

## Pairs Trading and Statistical Arbitrage

Pairs trading requires the spread between two assets to be stationary —
mean reversion around a stable level. Cointegration is tested with the
Engle-Granger procedure: regress one asset on the other and apply the
augmented dickey-fuller test to residuals (Johansen for multi-asset
baskets). Entry and exit trigger on the spread z-score (enter beyond 2,
exit near 0), with position sizing scaled to spread volatility. The
Ornstein-Uhlenbeck half-life (ln 2 / kappa) estimates holding period;
half-lives beyond weeks tie up capital and raise regime-break risk.
Stop-losses guard against structural breaks where cointegration fails.

## Credit Risk PD Modeling

Probability-of-default models for retail credit traditionally use logistic
regression on woe transformation features (weight of evidence bins),
selected by information value, yielding an interpretable credit score card
mapping log-odds to points. Discrimination is measured by roc-auc (Gini =
2*AUC-1); calibration maps scores to observed default rates. Defaults are
rare, so imbalanced classes are handled with class weights and threshold
tuning — while preserving calibration. Gradient boosting lifts AUC by
several points but requires explainability (SHAP, monotonic constraints)
for regulatory acceptance (SR 11-7, IFRS 9 staging).

## High-Yield Bond Spread Analysis

Corporate bond relative value uses option-adjusted spread: the constant
spread over the risk-free curve equating model and market price after
valuing embedded calls on a lattice — essential in high yield where
callability is pervasive. Reduced-form models express spread through the
default hazard rate times loss-given-default; hazard curves bootstrap from
CDS or bond strips. Portfolio construction applies duration matching to
isolate credit views from rate moves, monitors convexity (negative near
call prices), and separates a liquidity premium component (bid-ask, TRACE
volume) from pure credit spread when judging cheapness.

## Optimal Trade Execution

Almgren-Chriss decomposes market impact: permanent impact shifts the
equilibrium price with cumulative volume; temporary impact is a per-slice
cost of demanding liquidity that decays after each child order. The optimal
schedule over the liquidation horizon minimizes expected cost plus a
risk aversion parameter times cost variance — higher urgency front-loads
execution, lower urgency approaches TWAP. Benchmarks: vwap for
participation strategies, arrival-price (implementation shortfall) for
opportunity-cost accounting. Modern engines adapt via real-time signals
(spread, depth, short-term alpha) within the Almgren-Chriss frontier.

## Automated Market Makers (AMMs)

Constant-product AMMs quote via xy=k: any trade moves along the curve, so
price impact is deterministic and slippage tolerance bounds execution
price. Liquidity providers earn the liquidity provider fee (e.g. 0.3
percent) but suffer impermanent loss when the pool price diverges — the
loss versus holding grows with price divergence and is realized through
arbitrage rebalancing by external traders who restore the pool to market
price. Concentrated liquidity (Uniswap v3) lets LPs allocate capital to
price ranges, multiplying fee capture per dollar at the cost of more
active management and sharper divergence exposure.

## Inflation Forecasting with VARs

Macroeconomic inflation forecasting uses vector autoregressions over
core cpi or the pce deflator (the Fed's preferred gauge), activity, and
policy rates. lag order selection uses AIC/BIC under a parsimony prior;
over-parameterization is tamed with Bayesian shrinkage (Minnesota prior).
An impulse response function traces inflation's path after identified
shocks (Cholesky ordering or sign restrictions); granger causality tests
whether candidate indicators (wages, inflation expectations, supply-chain
pressure) add predictive content. Benchmarks to beat include random walk
and Phillips-curve models; structural breaks (post-2020) motivate
time-varying-parameter VARs.
