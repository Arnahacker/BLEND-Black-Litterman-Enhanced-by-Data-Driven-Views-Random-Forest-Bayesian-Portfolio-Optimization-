# Notes

## Theory and Assumptions Behind the Chosen Model

For my portfolio optimization, I combined two methods: **Random Forest**, which is a machine learning technique that looks for patterns in data, and the **Black-Litterman model**, a popular finance method that helps blend your own ideas with what the market expects.  

The basic idea is to use data to make smart guesses about future returns, then mix those guesses carefully with general market beliefs to build a balanced portfolio.  

This approach assumes that past patterns—like how quickly prices move up or down (**momentum**) or how much they vary (**volatility**)—can help predict what will happen next.  

Machine learning models like Random Forest can find these patterns if trained well. It’s also important to know how confident we are in these predictions because some guesses are more reliable than others.  

By combining our predictions with the market’s views through Black-Litterman, the portfolio avoids risky bets and stays reasonable. Since markets change over time, I used recent data to keep the model updated.  

---

## Why I Chose This Model

Before picking this method, I looked at many investment models.  

- **Modern Portfolio Theory (MPT)** tries to balance how much risk you take with how much return you get but depends on having very accurate inputs, which is tough.  
- **CAPM (Capital Asset Pricing Model)** explains how asset prices relate to market risk but assumes everyone has the same information and behaves the same way, which isn’t realistic.  
- **Black-Litterman** improves on MPT by mixing market views with your own, but it’s hard to create good personal views.  
- **Risk Parity** and **Hierarchical Risk Parity** aim to spread risk evenly but can sometimes favor very safe investments or rely too much on grouping similar assets.  
- **Minimum Variance** tries only to reduce risk but can ignore how much money you actually make.  
- **Sharpe Ratio Maximization** focuses on the best return per unit of risk but depends heavily on accurate predictions.  
- **Arbitrage Pricing Theory (APT)** uses many risk factors but needs a lot of data.  

I chose to combine **Random Forest** and **Black-Litterman** because Random Forest can find complicated patterns and create predictions (called “views”), while Black-Litterman mixes those views with market expectations to avoid overconfident bets.  

---

## How the Model Works

I used monthly returns over many years. Every month, the Random Forest model trains on the last **60 months (5 years)** of data to predict the return for the next month for each industry.  

I looked at two main features: **momentum** and **volatility**.  
In the future, I might add more features like **skewness** or **kurtosis**, which describe how returns behave.  

For each prediction, I measured how confident the model was by checking how well it did on recent test data—if it predicted well before, we trust it more now.  

These predictions became “views” in Black-Litterman, where the confidence scores decide how much trust to give each prediction.  

Then, the model combines these views with what the market expects using Black-Litterman’s math.  

Finally, I use this combined forecast and risk information to pick portfolio weights (how much money to put in each industry) that **maximize the Sharpe ratio**, which measures how much return you get for each unit of risk.  

---

## Strengths and Weaknesses

This model adapts well to changing market conditions by using recent data and makes investment decisions based on real patterns, not just guesses.  

The Black-Litterman method balances personal views with the overall market, helping avoid extreme investments.  

However, in combining Random Forest predictions with the Black-Litterman model, several challenges arose:  

1. Accurately estimating the confidence in the Random Forest views was difficult since prediction errors can be noisy and misleading.  
2. Random Forest captures **non-linear** patterns, while Black-Litterman assumes **linear** relationships, leading to potential mismatches.  
3. Covariance estimates used in Black-Litterman could be unstable due to limited data or changing markets.  
4. Practical issues like **transaction costs** and **market regime shifts** were not fully addressed.  

These challenges highlight the need for careful tuning and validation when blending machine learning with traditional portfolio optimization.  

---

## What I Can Do to Make It Better

To improve the model, I could:  

- Add more information like **economic trends** or **market sentiment**.  
- Use **cross-validation** to ensure reliability across different data samples.  
- Tune Random Forest hyperparameters for sharper predictions.  
- Use models that better capture **prediction uncertainty**.  
- Test the model in **extreme market conditions** (crashes, rallies).  

To prevent extreme model outputs, I **capped volatility predictions at 200%**, since higher values distorted risk assessment and produced unreliable decisions.  

---

## Something I Learned

During portfolio backtesting, I encountered a recurring error:

This occurred during the **covariance matrix** calculation step using `scikit-learn`.  

Even though the dataset (`train_data_bl`) appeared clean, the error originated from deep within scikit-learn’s C-level code, likely due to **near-perfect correlation** between asset returns, which causes **numerical instability** during matrix inversion.  

To stabilize the process, I introduced two solutions:

1. **Added Tiny Random Jitter (1e-10)** to the Data  
   - Introduced small random noise to break exact correlations.  
2. **Implemented a Covariance Estimation Fallback**  
   - Used **Ledoit-Wolf shrinkage** as default and **OAS (Oracle Approximating Shrinkage)** as backup.  

Both methods belong to the family of **shrinkage estimators**, which improve stability when dealing with limited data or many assets.  

---

## Resources Used (Outside of Those Provided)

- [Investopedia – Covariance](https://www.investopedia.com/terms/c/covariance.asp)  
- [Wikipedia – Floating Point Arithmetic](https://en.wikipedia.org/wiki/Floating-point_arithmetic)  
- [Scikit-learn – Covariance Estimation](https://scikit-learn.org/stable/modules/covariance.html)  
- [The Elements of Statistical Learning (Stanford)](https://web.stanford.edu/~hastie/ElemStatLearn/)  
- [ScienceDirect – Portfolio Optimization Study](https://www.sciencedirect.com/science/article/abs/pii/S0927539803000845)  
- [BSIC – Hierarchical Risk Parity](https://bsic.it/advanced-portfolio-optimization-hrp-hierarchical-risk-parity/)  
- [QuantInsti – Random Forest in Python](https://blog.quantinsti.com/random-forest-algorithm-in-python/)  
- [W3Schools – Pandas CSV](https://www.w3schools.com/python/pandas/pandas_csv.asp)  
- [YouTube: Portfolio Optimization Tutorials](https://www.youtube.com/watch?v=bDhvCp3_lYw)  
- [YouTube: Machine Learning in Finance](https://www.youtube.com/watch?v=mELtchoKXtM)  
- [YouTube: Black-Litterman Explained](https://www.youtube.com/watch?v=qNRODjhEDUA)  
- [Investopedia – Black-Litterman Model](https://www.investopedia.com/terms/b/black-litterman_model.asp)  

---

### Summary of Key Fixes
1. Added Tiny Random Jitter (1e-10) to the Data  
2. Implemented a Covariance Estimation Fallback  
