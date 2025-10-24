# Black-Litterman Portfolio Optimization with Random Forest Views


# ## 1. Importing Libraries and Initial Setup
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import expected_returns, risk_models
import logging
import matplotlib.pyplot as plt
import cvxpy as cp # Required by PyPortfolioOpt for optimization
import warnings
import math
import traceback

# Configure logging to suppress INFO messages from libraries like PyPortfolioOpt
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

# Suppress specific warnings that might clutter the output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='pypfopt')
warnings.filterwarnings('ignore', category=RuntimeWarning) # Suppress RuntimeWarning from numpy/sklearn for cleaner output

print("Libraries imported and logging configured.")


file_path = 'ind30_m_vw_rets (1).csv' # Adjust this path if your file is elsewhere

# Define column names: "Month" followed by 30 industry columns
column_names = ["Month"] + [f"Industry_{i+1}" for i in range(30)]

# Load the CSV file, skipping the header row (which is the first row)
dfmonthlyvalue = pd.read_csv(file_path, names=column_names)

# Drop the original header row (now at index 0) and reset the index
dfmonthlyvalue = dfmonthlyvalue.drop(index=0).reset_index(drop=True)

# Convert industry return columns to float and then to decimal format (divide by 100)
industry_cols = dfmonthlyvalue.columns[1:]
dfmonthlyvalue[industry_cols] = dfmonthlyvalue[industry_cols].astype(float) / 100

# Convert the 'Month' column to integer for easier datetime conversion
dfmonthlyvalue['Month'] = dfmonthlyvalue['Month'].astype(int)

# Convert the 'Month' column to datetime objects
dfmonthlyvalue['Month'] = pd.to_datetime(dfmonthlyvalue['Month'].astype(str).str.strip(), format='%Y%m', errors='coerce')

# Remove rows where 'Month' conversion failed (NaT values)
dfmonthlyvalue.dropna(subset=['Month'], inplace=True)

# Sort the DataFrame by 'Month' to ensure chronological order
dfmonthlyvalue.sort_values('Month', inplace=True)

# Set 'Month' as the DataFrame index
dfmonthlyvalue.set_index('Month', inplace=True)

# Filter the DataFrame to include data from January 1, 2000, onwards
dfmonthlyvalue_final = dfmonthlyvalue.loc['2000-01-01':].copy()

# Ensure all columns are numeric, coercing any non-numeric values to NaN
dfmonthlyvalue_final = dfmonthlyvalue_final.apply(pd.to_numeric, errors='coerce')

# Replace any infinite values with NaN
dfmonthlyvalue_final = dfmonthlyvalue_final.replace([np.inf, -np.inf], np.nan)

# Drop any rows that still contain NaN values after previous cleaning steps
dfmonthlyvalue_final.dropna(how="any", inplace=True)

print(f"Data loaded and preprocessed. Final data shape: {dfmonthlyvalue_final.shape}")
print("First 5 rows of final preprocessed data:")
print(dfmonthlyvalue_final.head())
print("\nDescription of final preprocessed data:")
print(dfmonthlyvalue_final.describe())

# ---
# ## 3. Rolling Window Backtesting Setup
# Define the size of the training window in months (e.g., 60 months = 5 years)
window_size = 60 # 60 months of historical data for training

# Define the horizon for which predictions are made (e.g., 1 month ahead)
test_horizon = 1 # Predict returns for the next month

# Determine the starting and ending points for the rolling window
start = window_size
end = len(dfmonthlyvalue_final) - test_horizon

# Lists to store results from each iteration of the backtest
weights_over_time = []
dates_list = []
performance_stats = []
portfolio_returns_actual = []

print(f"Backtesting parameters set: Window size = {window_size} months, Test horizon = {test_horizon} month.")
print(f"Backtesting will run from index {start} to {end-1} (Total iterations: {end - start}).")

# ---
# ## 4. Rolling Window Backtesting Loop (with enhanced debugging and robustness)
print("Starting rolling window backtesting loop...")
for current_end in range(start, end):
    current_date = dfmonthlyvalue_final.index[current_end]

    # --- DEBUGGING BLOCK FOR 2005-11 ---
    # Set DEBUG_MODE to True for the problematic iteration, False otherwise.
    # This will print extensive details for that specific month.
    if current_date.strftime('%Y-%m') == '2005-11':
        print(f"\n--- DEBUGGING Iteration {current_end - start + 1}/{end - start} (Date: {current_date.strftime('%Y-%m')}) ---")
        DEBUG_MODE = True
    else:
        DEBUG_MODE = False
        if not DEBUG_MODE: # Only print general iteration info if not in specific debug mode
            print(f"\n--- Iteration {current_end - start + 1}/{end - start} (Date: {current_date.strftime('%Y-%m')}) ---")
    # --- END DEBUGGING BLOCK SETUP ---

    try:
        # Define the training data window
        train_data = dfmonthlyvalue_final.iloc[current_end - window_size : current_end].copy()

        if DEBUG_MODE:
            print("\n--- DEBUG: After initial train_data selection ---")
            if train_data.isnull().any().any():
                print(f"DEBUG: NaNs found in train_data BEFORE column drop:\n{train_data.isnull().sum()[train_data.isnull().sum() > 0]}")
            if (train_data.abs() > 10).any().any():
                 print(f"DEBUG: Extreme values (>1000%) found in train_data BEFORE column drop:\n{train_data[train_data.abs() > 10].stack()}")

        train_data.dropna(axis=1, how='any', inplace=True)
        if train_data.empty:
            logging.warning(f"Train data became empty after dropping NaN columns at {current_date.strftime('%Y-%m')}. Skipping.")
            continue
        if DEBUG_MODE:
            print(f"DEBUG: Train data shape after NaN column drop: {train_data.shape}")

        if train_data.empty or len(train_data) < window_size * 0.5:
            logging.warning(f"Train data too small or empty at {current_date.strftime('%Y-%m')}. Skipping.")
            continue

        rebalancing_date = dfmonthlyvalue_final.index[current_end -1]
        test_date = dfmonthlyvalue_final.index[current_end + test_horizon - 1]

        # --- Feature Engineering for Random Forest Model ---
        features = pd.DataFrame(index=train_data.index)
        features['momentum'] = train_data.mean(axis=1).rolling(3).mean()
        features['volatility'] = train_data.std(axis=1).rolling(3).mean()
        features['skew'] = train_data.skew(axis=1).rolling(3).mean()
        features['kurtosis'] = train_data.kurtosis(axis=1).rolling(3).mean()
        features.dropna(inplace=True)

        if features.empty:
            logging.warning(f"Features DataFrame is empty after calculation at {current_date.strftime('%Y-%m')}. Skipping.")
            continue

        train_data_aligned = train_data.loc[features.index]
        if train_data_aligned.empty:
            logging.warning(f"Aligned train data for RF is empty at {current_date.strftime('%Y-%m')}. Skipping.")
            continue
        if DEBUG_MODE:
            print(f"DEBUG: Train data aligned for RF (shape): {train_data_aligned.shape}")

        # --- Random Forest for Views and Confidences ---
        views = {}
        raw_confidences = {}

        for asset in train_data_aligned.columns:
            y = train_data_aligned[asset].shift(-test_horizon).dropna()
            X = features.loc[y.index]

            if X.empty or y.empty or len(X) < 5: # Ensure at least 5 data points for RF training
                logging.debug(f"Not enough data for RF for asset {asset} at {current_date.strftime('%Y-%m')}. Skipping asset.")
                continue

            split_idx = int(len(X) * 0.75)
            if split_idx < 1 or len(X) - split_idx < 1: # Ensure both train and test sets have at least 1 sample
                logging.debug(f"Split too small for RF for asset {asset} at {current_date.strftime('%Y-%m')}. Skipping asset.")
                continue

            X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
            X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_current_period_features = scaler.transform(X.tail(1))

            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train_scaled, y_train)

            prediction = model.predict(X_current_period_features)[0]
            views[asset] = prediction

            y_pred_test = model.predict(X_test_scaled)
            error = mean_squared_error(y_test, y_pred_test)
            confidence = 1 / (1 + error)
            raw_confidences[asset] = confidence

        if len(views) == 0:
            logging.warning(f"No views generated for any asset at {current_date.strftime('%Y-%m')}. Skipping.")
            continue
        if DEBUG_MODE:
            print(f"DEBUG: Generated views for {len(views)} assets.")

        # --- Black-Litterman Model Inputs Preparation ---
        train_data_bl = train_data.copy()

        # Robustness Step 1: Explicitly handle inf and NaN before statistical calculations
        if DEBUG_MODE:
            print("\n--- DEBUG: Before initial cleaning train_data_bl for covariance ---")
            if train_data_bl.isnull().any().any():
                print(f"DEBUG: NaNs found in train_data_bl (before replace/dropna):\n{train_data_bl.isnull().sum()[train_data_bl.isnull().sum() > 0]}")
            if np.isinf(train_data_bl).any().any():
                print(f"DEBUG: Infs found in train_data_bl (before replace/dropna):\n{train_data_bl[np.isinf(train_data_bl)].stack()}")
            if (train_data_bl.abs() > 5).any().any():
                print(f"DEBUG: Extreme values (>500%) found in train_data_bl (before replace/dropna):\n{train_data_bl[train_data_bl.abs() > 5].stack()}")

        train_data_bl.replace([np.inf, -np.inf], np.nan, inplace=True)
        train_data_bl.dropna(how='any', axis=0, inplace=True) # Drop rows with any NaNs that might have been introduced

        # Robustness Step 2: More aggressive clipping for extremely large return values
        clip_value = 5.0 # Max absolute return of 500% (e.g., 5.0 for monthly)
        train_data_bl = train_data_bl.clip(lower=-clip_value, upper=clip_value)

        # --- NEW ADDITION: Add a tiny amount of jitter for numerical stability ---
        jitter_amount = 1e-10 # A very small number
        train_data_bl = train_data_bl + np.random.normal(0, jitter_amount, train_data_bl.shape)

        # Drop columns with near-zero standard deviation (or all zeros)
        stds = train_data_bl.std()
        train_data_bl = train_data_bl.drop(columns=stds[stds < 1e-6].index) # Using 1e-6 for "near-zero"

        # FINAL CHECK ON train_data_bl BEFORE COVARIANCE
        if DEBUG_MODE:
            print("\n--- DEBUG: FINAL train_data_bl state BEFORE Covariance Calculation ---")
            if train_data_bl.empty:
                print("DEBUG: train_data_bl is EMPTY!")
            print(f"DEBUG: train_data_bl shape: {train_data_bl.shape}")
            print(f"DEBUG: train_data_bl head:\n{train_data_bl.head()}")
            print(f"DEBUG: train_data_bl describe:\n{train_data_bl.describe()}")

            # Check for NaNs and print their exact locations (already there, keep it)
            nan_present = train_data_bl.isnull().any().any()
            if nan_present:
                print(f"DEBUG: NaNs found in train_data_bl BEFORE COVARIANCE:")
                # Print columns with NaNs and their counts
                print(train_data_bl.isnull().sum()[train_data_bl.isnull().sum() > 0])
                # Print specific rows/columns with NaNs
                nan_mask = train_data_bl.isnull()
                for col in train_data_bl.columns:
                    if nan_mask[col].any():
                        nan_indices = train_data_bl.index[nan_mask[col]].tolist()
                        print(f"  Asset: {col} has NaNs at dates: {nan_indices}")


            # Check for Infs and print their exact locations (already there, keep it)
            inf_present = np.isinf(train_data_bl).any().any()
            if inf_present:
                print(f"DEBUG: Infs found in train_data_bl BEFORE COVARIANCE:")
                inf_mask = np.isinf(train_data_bl)
                for col in train_data_bl.columns:
                    if inf_mask[col].any():
                        inf_indices = train_data_bl.index[inf_mask[col]].tolist()
                        print(f"  Asset: {col} has Infs at dates: {inf_indices}")

            if not nan_present and not inf_present:
                print("DEBUG: train_data_bl contains NO NaNs or Infs before covariance calculation. This is unexpected given the error.")

            # --- NUMERICAL STABILITY CHECKS for train_data_bl ---
            print("\n--- DEBUG: NUMERICAL STABILITY CHECKS for train_data_bl ---")
            # 1. Check for columns with very low/zero standard deviation
            stds_after_drop = train_data_bl.std()
            zero_std_cols = stds_after_drop[stds_after_drop < 1e-9].index.tolist() # Using 1e-9 for even smaller
            if zero_std_cols:
                print(f"DEBUG: Columns with near-zero standard deviation (potentially causing issues): {zero_std_cols}")
            else:
                print("DEBUG: No columns with near-zero standard deviation detected.")

            # 2. Check the condition number of the correlation matrix
            try:
                corr_matrix = train_data_bl.corr()
                corr_matrix.fillna(0, inplace=True) # Fill NaNs that occur from constant columns with 0 correlation
                condition_number = np.linalg.cond(corr_matrix)
                print(f"DEBUG: Condition number of correlation matrix: {condition_number}")
                if condition_number > 1e10: # Threshold for concern varies, 1e10 is usually very high
                    print("DEBUG: WARNING: High condition number indicates near-singular matrix! This is likely the cause.")
            except np.linalg.LinAlgError as e:
                print(f"DEBUG: Could not compute condition number (Singular matrix error): {e}")
                print("DEBUG: This strongly suggests a singular matrix issue.")
            except Exception as e:
                print(f"DEBUG: Error computing correlation matrix or condition number: {e}")

            # 3. Check for duplicate rows (perfectly correlated time series)
            if train_data_bl.duplicated().any().any(): # Use .any().any() for DataFrame
                print("DEBUG: WARNING: Duplicate rows found in train_data_bl. This could lead to numerical issues.")
            else:
                print("DEBUG: No duplicate rows found.")

            # 4. Check for constant columns (all values are the same)
            constant_cols = [col for col in train_data_bl.columns if train_data_bl[col].nunique() == 1]
            if constant_cols:
                print(f"DEBUG: WARNING: Constant columns detected: {constant_cols}. These will cause issues.")
            else:
                print("DEBUG: No constant columns detected.")

            # 5. Check if any columns became constant after clipping (less common, but possible)
            if DEBUG_MODE:
                for col in train_data_bl.columns:
                    if train_data_bl[col].std() < 1e-9: # Re-check std after clipping
                        print(f"DEBUG: Column '{col}' has very low std ({train_data_bl[col].std():.2e}) after clipping.")
            # --- END NUMERICAL STABILITY CHECKS ---


        if train_data_bl.empty or train_data_bl.shape[1] < 2:
            logging.warning(f"train_data_bl became empty or has less than 2 columns after cleaning at {current_date.strftime('%Y-%m')}. Skipping.")
            continue
        print(f"train_data_bl shape for mu/S calculation (after all pre-S cleaning): {train_data_bl.shape}")

        mu = expected_returns.mean_historical_return(train_data_bl, compounding=False)

        # Robustness Step 3: Try-except for covariance calculation, with multiple fallbacks
        S = None # Initialize S to None
        try:
            S = risk_models.CovarianceShrinkage(train_data_bl).ledoit_wolf()
            if S.isnull().any().any() or np.isinf(S).any().any():
                raise ValueError("Ledoit-Wolf covariance matrix contains NaNs or Infs after calculation.")
            if DEBUG_MODE:
                print("DEBUG: Ledoit-Wolf S calculated successfully.")
        except Exception as e:
            logging.warning(f"Ledoit-Wolf covariance failed at {current_date.strftime('%Y-%m')}: {e}. Attempting OAS covariance.")
            if DEBUG_MODE:
                traceback.print_exc() # Print traceback for Ledoit-Wolf failure

            try:
                # Fallback 1: Oracle Approximating Shrinkage
                S = risk_models.CovarianceShrinkage(train_data_bl).oas()
                if S.isnull().any().any() or np.isinf(S).any().any():
                    raise ValueError("OAS covariance also contains NaNs or Infs.")
                if DEBUG_MODE:
                    print("DEBUG: OAS S calculated successfully.")
            except Exception as inner_e:
                logging.warning(f"OAS covariance failed at {current_date.strftime('%Y-%m')}: {inner_e}. Attempting simple historical covariance.")
                if DEBUG_MODE:
                    traceback.print_exc() # Print traceback for OAS failure

                try:
                    # Fallback 2: Sample Covariance (if OAS also fails)
                    S = risk_models.sample_cov(train_data_bl)
                    if S.isnull().any().any() or np.isinf(S).any().any():
                        raise ValueError("Sample covariance also contains NaNs or Infs.")
                    if DEBUG_MODE:
                        print("DEBUG: Sample covariance S calculated successfully.")
                except Exception as deepest_e:
                    logging.error(f"Even sample covariance failed at {current_date.strftime('%Y-%m')}: {deepest_e}. Skipping iteration.")
                    if DEBUG_MODE:
                        traceback.print_exc() # Print traceback for sample_cov failure
                    continue # Skip this iteration entirely if covariance cannot be computed reliably

        # Final check on S (after potential fallback)
        if S is None or S.isnull().any().any() or np.isinf(S).any().any():
            logging.error(f"Covariance Matrix (S) is still invalid (NaNs/Infs) at {current_date.strftime('%Y-%m')}. Skipping.")
            continue

        if DEBUG_MODE:
            print(f"DEBUG: Final S shape: {S.shape}")
            print(f"DEBUG: Final S head:\n{S.head()}")


        valid_assets_for_bl = list(set(mu.index) & set(S.columns) & set(views.keys()) & set(raw_confidences.keys()))

        if len(valid_assets_for_bl) < 2:
            logging.warning(f"Not enough valid assets for Black-Litterman at {current_date.strftime('%Y-%m')}. Skipping.")
            continue

        mu_aligned = mu.loc[valid_assets_for_bl]
        S_aligned = S.loc[valid_assets_for_bl, valid_assets_for_bl]
        views_aligned = {k: views[k] for k in valid_assets_for_bl}
        raw_confidences_aligned = pd.Series(raw_confidences).loc[valid_assets_for_bl]

        min_conf = raw_confidences_aligned.min()
        max_conf = raw_confidences_aligned.max()
        if np.isclose(max_conf, min_conf):
            norm_confidences = pd.Series(0.5, index=raw_confidences_aligned.index)
        else:
            norm_confidences = (raw_confidences_aligned - min_conf) / (max_conf - min_conf)

        epsilon = 1e-4
        confidences_aligned = np.clip(norm_confidences.values, epsilon, 1.0)

        max_abs_view = max((abs(v) for v in views_aligned.values()), default=0.0)
        scale_factor = 0.05 / max_abs_view if max_abs_view > 0.05 else 1.0
        views_scaled = {k: v * scale_factor for k, v in views_aligned.items()}

        mu_clipped = mu_aligned.clip(lower=-0.05, upper=0.05)

        # Final check for NaNs before Black-Litterman model
        if mu_clipped.isna().any() or S_aligned.isna().any().any() or np.isnan(confidences_aligned).any() or any(np.isnan(v) for v in views_scaled.values()):
            print(f"DEBUG: NaNs found in BL final inputs at {current_date.strftime('%Y-%m')}. Skipping.")
            raise ValueError("Input contains NaN for Black-Litterman model.")

        # --- Black-Litterman Model Execution ---
        bl = BlackLittermanModel(
            S_aligned,
            pi=mu_clipped,
            absolute_views=views_scaled,
            view_confidences=confidences_aligned
        )
        bl_mu = bl.bl_returns()

        bl_mu = bl_mu.replace([np.inf, -np.inf], np.nan).dropna()
        if bl_mu.empty:
            logging.warning(f"BL returns are empty after cleaning at {current_date.strftime('%Y-%m')}. Skipping.")
            continue

        # --- Portfolio Optimization (Max Sharpe) ---
        cleaned_weights = {}
        solvers = ['ECOS', 'SCS', 'OSQP']
        for solver in solvers:
            try:
                assets_for_ef = list(set(bl_mu.index) & set(S_aligned.columns))
                if len(assets_for_ef) < 2:
                    raise ValueError("Not enough common assets for Efficient Frontier.")

                bl_mu_ef = bl_mu.loc[assets_for_ef]
                S_ef = S_aligned.loc[assets_for_ef, assets_for_ef]

                ef = EfficientFrontier(bl_mu_ef, S_ef, solver=solver)
                weights = ef.max_sharpe()
                cleaned_weights = ef.clean_weights()
                break
            except Exception as e:
                logging.warning(f"Optimization with solver {solver} failed at {current_date.strftime('%Y-%m')}: {e}")
                continue
        else:
            logging.warning(f"All solvers failed for portfolio optimization at {current_date.strftime('%Y-%m')}. Skipping.")
            continue

        if not cleaned_weights:
            logging.warning(f"No weights generated after optimization at {current_date.strftime('%Y-%m')}. Skipping.")
            continue

        # --- Store Weights ---
        all_original_assets = dfmonthlyvalue_final.columns.tolist()
        full_weights = {asset: cleaned_weights.get(asset, 0.0) for asset in all_original_assets}
        weights_over_time.append(full_weights)
        dates_list.append(rebalancing_date)

        # --- Calculate Portfolio Performance Metrics for the Current Period's Forecast ---
        w_series = pd.Series(cleaned_weights)
        common_assets_for_performance = list(set(w_series.index) & set(bl_mu_ef.index) & set(S_ef.columns))
        if len(common_assets_for_performance) < 1:
            logging.warning(f"Not enough common assets for performance calculation at {current_date.strftime('%Y-%m')}. Skipping performance stats.")
            continue

        w_vector_perf = w_series.loc[common_assets_for_performance].values
        bl_mu_perf = bl_mu_ef.loc[common_assets_for_performance].values
        S_perf = S_ef.loc[common_assets_for_performance, common_assets_for_performance].values

        port_return = np.dot(w_vector_perf, bl_mu_perf)
        port_volatility = math.sqrt(np.dot(w_vector_perf.T, np.dot(S_perf, w_vector_perf)))

        risk_free_rate = 0.01

        port_return_annual = np.clip(port_return * 12, -1.0, 1.0)
        port_vol_annual = np.clip(port_volatility * np.sqrt(12), 1e-4, 2.0)

        sharpe_ratio = (port_return_annual - risk_free_rate) / port_vol_annual if port_vol_annual > 1e-6 else np.nan

        performance_stats.append({
            'date': rebalancing_date,
            'expected_return_annual': port_return_annual,
            'volatility_annual': port_vol_annual,
            'sharpe_ratio': sharpe_ratio
        })

        # --- Calculate Actual Out-of-Sample Portfolio Returns ---
        if test_date in dfmonthlyvalue_final.index:
            next_month_actual_returns = dfmonthlyvalue_final.loc[test_date]
            actual_weighted_return = 0.0
            for asset, weight in full_weights.items():
                asset_actual_return = next_month_actual_returns.get(asset, 0)
                actual_weighted_return += asset_actual_return * weight
            portfolio_returns_actual.append({'date': test_date, 'actual_return': actual_weighted_return})
            print(f"  Actual Return ({test_date.strftime('%Y-%m')}): {actual_weighted_return:.6f}")
        else:
            logging.warning(f"Actual returns for {test_date.strftime('%Y-%m')} not found. Cannot calculate out-of-sample return.")
            portfolio_returns_actual.append({'date': test_date, 'actual_return': np.nan})


    except Exception as e:
        logging.error(f"Skipping iteration {current_end} ({current_date.strftime('%Y-%m')}) due to error: {e}")
        traceback.print_exc() # Print full traceback for deep debugging
        fallback_date = dfmonthlyvalue_final.index[current_end -1] if current_end > 0 else pd.Timestamp('2000-01-01')

        dates_list.append(fallback_date)
        weights_over_time.append({asset: 0.0 for asset in dfmonthlyvalue_final.columns})
        performance_stats.append({
            'date': fallback_date,
            'expected_return_annual': np.nan,
            'volatility_annual': np.nan,
            'sharpe_ratio': np.nan
        })
        portfolio_returns_actual.append({'date': test_date if 'test_date' in locals() else fallback_date, 'actual_return': np.nan})
        continue

print("\nBacktesting loop finished.")

# ---
# ## 5. Results Processing and Visualization
print("Processing and visualizing results...")

# Convert list of weight dictionaries to a DataFrame
weights_df = pd.DataFrame(weights_over_time, index=dates_list)
weights_df.index.name = 'Date'
print("\n--- Portfolio Weights Summary ---")
print("Top 5 rows of calculated weights:")
print(weights_df.head())
print("\nMean weights across all assets:")
print(weights_df.mean().sort_values(ascending=False).head(10)) # Top 10 average weights

# Plotting the weights of the top 5 assets over time (based on average weight)
if not weights_df.empty:
    if len(weights_df.columns) > 0 and not weights_df.mean().sort_values(ascending=False).empty:
        top_assets = weights_df.mean().sort_values(ascending=False).head(5).index
        if not top_assets.empty:
            plt.figure(figsize=(12, 6))
            weights_df[top_assets].plot(ax=plt.gca())
            plt.title("Top 5 Asset Weights Over Time")
            plt.xlabel("Date")
            plt.ylabel("Weight")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print("No top assets to plot (weights_df might contain only zeros).")
    else:
        print("Weights DataFrame has no valid asset columns to plot.")
else:
    print("Weights DataFrame is empty. Cannot plot weights.")


# Convert list of performance statistics to a DataFrame
performance_df = pd.DataFrame(performance_stats).set_index('date')
performance_df.index.name = 'Date'
print("\n--- Portfolio Performance Summary (Based on Expected Returns) ---")
print(performance_df.head())
print("\nOverall performance statistics:")
print(performance_df.describe())

# Plotting portfolio performance metrics over time
if not performance_df.empty:
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    performance_df['expected_return_annual'].plot(ax=axes[0], title='Annualized Expected Portfolio Return', color='blue')
    axes[0].set_ylabel("Return")
    axes[0].grid(True)

    performance_df['volatility_annual'].plot(ax=axes[1], title='Annualized Expected Portfolio Volatility', color='red')
    axes[1].set_ylabel("Volatility")
    axes[1].grid(True)

    performance_df['sharpe_ratio'].plot(ax=axes[2], title='Annualized Expected Portfolio Sharpe Ratio', color='green')
    axes[2].set_ylabel("Sharpe Ratio")
    axes[2].set_xlabel("Date")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()
else:
    print("Performance DataFrame is empty. Cannot plot performance metrics.")

# Convert actual portfolio returns to a DataFrame
actual_returns_df = pd.DataFrame(portfolio_returns_actual).set_index('date')
actual_returns_df.index.name = 'Date'
print("\n--- Out-of-Sample Actual Portfolio Returns ---")
print(actual_returns_df.head())
print("\nDescriptive statistics for actual returns:")
print(actual_returns_df.describe())

# Calculate cumulative actual returns
if not actual_returns_df.empty and 'actual_return' in actual_returns_df.columns:
    cumulative_returns = (1 + actual_returns_df['actual_return'].fillna(0)).cumprod() - 1
    plt.figure(figsize=(12, 6))
    cumulative_returns.plot(title='Cumulative Out-of-Sample Portfolio Returns')
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("Actual returns DataFrame is empty or 'actual_return' column missing. Cannot plot cumulative returns.")

print("\nAnalysis complete.")
