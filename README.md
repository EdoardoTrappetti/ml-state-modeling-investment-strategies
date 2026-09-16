# Machine Learning-Based State Modeling for Dynamic Investment Strategies

A machine learning framework for extracting **latent information states from macroeconomic data** and translating them into **dynamic asset allocation strategies**.

Developed as part of a Master's thesis in Quantitative Finance, the project investigates whether recurring structures in the macroeconomic information space can support systematic portfolio decisions without using financial-market variables to identify the states.

---

## Overview

The framework combines unsupervised learning, supervised classification and portfolio optimization in a unified investment pipeline:

**Macroeconomic Data → Feature Representation → State Identification → State Classification → Portfolio Optimization → Out-of-Sample Evaluation**

Two alternative representations of the macroeconomic information set are considered:

- **RAW** — standardized original variables
- **PCA** — lower-dimensional representation based on Principal Component Analysis

Latent information states are identified through **K-Means clustering**, while **Linear Discriminant Analysis (LDA)** is used in the baseline framework to translate the identified state structure into a probabilistic classification of new observations.

The resulting information is then connected to state-specific portfolio allocations.

---

## Methodology

The empirical pipeline consists of six main stages:

1. **Data preprocessing**  
   Collection, frequency alignment, transformation and standardization of macroeconomic variables.

2. **Feature representation**  
   Comparison between the original standardized feature space and a PCA-based representation.

3. **State identification**  
   K-Means clustering with alternative numbers of latent information states.

4. **State classification**  
   LDA classification of new macroeconomic observations in the baseline framework.

5. **Portfolio construction**  
   State-specific portfolios estimated using:
   - Maximum Sharpe Ratio
   - Mean-Variance Optimization

6. **Out-of-sample backtesting**  
   Dynamic portfolio implementation with transaction costs and multiple risk and performance diagnostics.

---

## Macroeconomic Information Set

State identification relies exclusively on macroeconomic and information variables rather than asset returns or market-volatility indicators.

The information set captures several dimensions of the economic environment, including:

- Energy prices and inflation
- Financial stress
- Bank lending conditions
- Global economic uncertainty
- Pandemic-related uncertainty
- Geopolitical risk
- Global supply-chain pressures

This separation is intended to keep the state-identification process distinct from the financial returns subsequently used for portfolio construction.

---

## Investment Framework

Each identified information state is associated with the historical behaviour of a diversified set of financial assets.

The investment universe covers:

- U.S. equities
- European equities
- Emerging markets
- Long-term U.S. Treasury bonds
- Gold
- Broad commodities
- Short-term U.S. Treasury bills

For each state, expected returns and covariance matrices are estimated from the corresponding historical asset returns and used to construct state-specific optimal portfolios.

The baseline allocation dynamically combines these portfolios according to the information produced by the classification stage.

---

## Model Configurations

The project evaluates different levels of model adaptivity.

### Static

The machine learning representation and state-specific portfolios are estimated on the training sample and remain fixed during the out-of-sample period.

### Expanding

The machine learning representation remains fixed, while state-specific return moments and portfolio allocations are progressively updated as new financial observations become available.

### Rolling PCA

The scaler, PCA representation and K-Means clustering structure are recursively re-estimated using an expanding macroeconomic information set.

### Annual Adaptive Re-estimation

Candidate model specifications are periodically re-estimated and evaluated using historically available information. Model selection is therefore allowed to evolve through time rather than remaining permanently tied to a single ex-ante specification.

---

## Out-of-Sample Evaluation

The framework is evaluated using a strict train/test structure with a one-period information lag between macroeconomic information and portfolio implementation.

Portfolio performance is evaluated net of **10 bps proportional transaction costs**.

The main evaluation metrics include:

- Annualized Return
- Annualized Volatility
- Sharpe Ratio
- Maximum Drawdown
- Calmar Ratio
- Value at Risk (VaR)
- Expected Shortfall (ES)
- CAPM Alpha and Beta
- Alpha t-statistic
- CAPM R²
- Portfolio Turnover

The strategies are compared with market and no-clustering benchmarks.