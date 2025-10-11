# 🧠 BLEND: Black-Litterman Enhanced by Data-Driven Views

**Author:** Arnav Chhabra  
**Description:**  
This project integrates **machine learning predictions** with the **Black-Litterman portfolio optimization framework** to create a hybrid quantitative strategy.  
By using **Random Forest regressors** to generate dynamic “views” on asset returns, and combining them with market equilibrium expectations via **Bayesian updating**, the model seeks to construct more adaptive, data-informed portfolios.

---

## 🚀 Overview

The **Black-Litterman model** traditionally blends investor views with market equilibrium returns to produce improved expected returns.  
In this implementation, the “investor views” are **learned automatically** through machine learning models (Random Forests) trained on engineered features from historical data.

---

## 🔍 Methodology

### 1. **Data Loading and Preprocessing**
- Input: Monthly industry returns (`ind30_m_vw_rets.csv`)
- Cleans and converts data to proper datetime and numeric formats.
- Filters for data post-2000.

### 2. **Feature Engineering**
Generates four time-series features from past returns:
- **Momentum:** Rolling mean of average returns.
- **Volatility:** Rolling standard deviation.
- **Skewness:** Rolling skewness of return distributions.
- **Kurtosis:** Rolling kurtosis of return distributions.

### 3. **Random Forest for Predictive Views**
- Trains a separate `RandomForestRegressor` per asset on historical data.
- Predicts **next-month returns**, which are treated as the model’s “views.”
- Confidence in each view is computed as a function of out-of-sample MSE:  
  \[
  \text{Confidence} = \frac{1}{1 + \text{MSE}}
  \]

### 4. **Black-Litterman Integration**
- Combines market equilibrium returns (`μ`) and covariances (`Σ`) with ML-derived views.
- View confidences are normalized and used as Bayesian weights.
- Produces posterior expected returns `μ*` through the **Black-Litterman formula**.

### 5. **Portfolio Optimization**
- Uses `PyPortfolioOpt`’s `EfficientFrontier` to compute:
  - Maximum Sharpe ratio portfolio
  - Cleaned and normalized weights
- Includes solver fallback (ECOS → SCS → OSQP) for robustness.

### 6. **Rolling Window Backtesting**
- Rolling 60-month training windows
- Predicts 1 month ahead per iteration
- Iteratively updates:
  - Model training
  - BL integration
  - Portfolio optimization
- Stores:
  - Portfolio weights
  - Expected annualized returns, volatilities, Sharpe ratios
  - Actual out-of-sample returns

### 7. **Performance Evaluation**
- Aggregates and visualizes:
  - Portfolio weights over time
  - Sharpe ratio trajectory
  - Out-of-sample portfolio returns

---

## ⚙️ Installation

### Prerequisites
Ensure the following libraries are installed:
```bash
pip install numpy pandas matplotlib scikit-learn PyPortfolioOpt cvxpy
