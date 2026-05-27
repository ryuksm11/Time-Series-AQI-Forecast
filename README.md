# Univariate AQI Time Series Forecasting

Weekly Air Quality Index (AQI) forecasting using classical statistical time series methods — SARIMA and ETS — with a statistics-first approach where every modelling decision is backed by a named diagnostic or test.

---

## Problem Statement

Forecast the weekly AQI for an Indian urban monitoring site across a 38-week horizon (April–December 2025) with point forecasts and 80%/95% prediction intervals. The data follows the India CPCB AQI scale (0–500), with a strong annual cycle driven by monsoon rainfall (summer trough) and winter thermal inversions (winter peak).

---

## Data

| Property | Value |
|---|---|
| File | `AQI_Univariate_TimeSeries.csv` |
| Raw frequency | Daily, 2017-01-01 → 2025-12-31 (3,287 rows) |
| Observed span | 2017-01-01 → 2025-03-31 (2,979 values) |
| Forecast horizon | 2025-04-01 → 2025-12-31 (275 daily rows, all NaN) |
| Interior missing values | 33 daily NaNs (genuine gaps) |
| AQI scale | Integer, 41–494 (CPCB India) |
| Modelling frequency | **Weekly** (aggregated from daily) |

**CPCB AQI bands:** Good (0–50) · Satisfactory (51–100) · Moderate (101–200) · Poor (201–300) · Very Poor (301–400) · Severe (401–500)

---

## Methodology Pipeline

```
Raw daily CSV
     │
     ▼
Stage 1 — Aggregation & imputation
  • Daily → weekly mean (resample('W'), week-ending Sunday)
  • Coverage threshold: ≥4 observed days per week
  • 5 weeks imputed (1.2%): deseasonalise → linear interpolate → reseasonalise
  • Boundary weeks: pure week-of-year climatology
     │
     ▼
Stage 2 — Stationarity & transformation
  • Variance check: Levene p=0.960 → homoskedastic → no Box-Cox
  • ADF (lags=12, AIC): p≈0 + KPSS p≥0.10 → d=0
  • STL seasonal strength Fs=0.875; variance drops 62.5% after D=1 → D=1
  • Final decision: d=0, D=1, s=52
     │
     ▼
Stage 3 — EDA & model identification
  • STL decomposition (additive, robust, s=52)
  • Seasonal strength Fs=0.875, trend strength Ft=0.113
  • ACF/PACF of D=1 series → PACF cuts off at lag 2 → AR(2)
  • Isolated ACF spike at lag 52 (−0.44) → SMA(1)
  • Candidate set: SARIMA(1,0,0)(0,1,1)₅₂, (2,0,0)(0,1,1)₅₂,
                   (1,0,0)(1,1,0)₅₂, (2,0,0)(1,1,0)₅₂
     │
     ▼
Stage 4 — Modelling & evaluation
  • Train/val split: 380 weeks train / 52 weeks validation (time-ordered)
  • Rolling-origin backtesting: 4 origins × 12 steps
  • Baselines: Naive, Seasonal Naive, Drift
  • ETS: Holt-Winters additive (undamped + damped)
  • SARIMA: all 4 candidates fitted, compared by AIC/BIC + validation MASE
  • Residual diagnostics: Ljung-Box (lags 10, 20, 52), Q-Q, Shapiro-Wilk
     │
     ▼
Stage 5 — Final forecast
  • Refit winner on full 432-week series
  • 38-week horizon: 2025-04-13 → 2025-12-28
  • Point forecast + 80%/95% prediction intervals
  • Output: CSV + forecast plot with CPCB band overlay
```

---

## Key Statistical Decisions

| Decision | Value | Justification |
|---|---|---|
| Aggregation | Weekly mean, Sunday-end | s=52 tractable for SARIMA; daily s=365 is not |
| Partial-week threshold | ≥ 4 observed days | Variance of estimate ≤ 1.75× full-week variance |
| Variance transform | None | Levene p=0.960 — homoskedastic across all years |
| Regular differencing | d = 0 | ADF p≈0 (lags=12); KPSS stat=0.034, p≥0.10 |
| Seasonal differencing | D = 1, s = 52 | Fs=0.875; 62.5% variance reduction; confirmed by ADF+KPSS |
| Decomposition type | Additive | Stable seasonal amplitude confirmed by Levene test |
| SARIMA orders | (2,0,0)(0,1,1)₅₂ | PACF cutoff lag 2 → AR(2); ACF spike lag 52 → SMA(1) |
| Model selection | SARIMA(2,0,0)(0,1,1)₅₂ | Lowest MASE (0.641); clean residuals at all lags |
| ETS disqualified | ETS(A,Ad,A) | Ljung-Box fails at lags 20 (p=0.047) and 52 (p=0.027) |

---

## Results

### Validation set (52 weeks, 2024-04-14 → 2025-04-06)

| Model | MAE | RMSE | MAPE | MASE | R² |
|---|---|---|---|---|---|
| Naive | 81.48 | 102.98 | 44.1% | 1.702 | −0.21 |
| Drift | 89.12 | 111.93 | 45.4% | 1.861 | −0.42 |
| Seasonal Naive | 39.69 | 49.17 | 22.3% | 0.829 | 0.73 |
| ETS(A,Ad,A) | 31.44 | 39.99 | 16.7% | 0.657 | 0.82 |
| **SARIMA(2,0,0)(0,1,1)₅₂** | **30.68** | **38.95** | **17.0%** | **0.641** | **0.83** |

SARIMA is **36% more accurate than Seasonal Naive** (MASE 0.641 vs 0.829) and explains **83% of the variance** in the validation year. Rolling-origin backtesting across 4 origins confirms the ranking is stable.

### Final forecast (38 weeks, 2025-04-13 → 2025-12-28)

| Period | Mean AQI | CPCB Band | Driver |
|---|---|---|---|
| Apr–May | ~200–212 | Moderate | Post-winter transition |
| Jun–Sep | ~92–158 | Satisfactory / Moderate | Monsoon rainfall |
| Oct | ~249 | Poor | Post-monsoon dry air |
| Nov–Dec | ~335–346 | Very Poor | Winter inversion + stubble burning |

95% prediction interval stabilises at ≈ ±89 AQI units by week 5 (bounded by seasonal climatology, not growing unboundedly as in a random walk).

---

## Project Structure

```
Time_Series_AQI_Forecast/
├── aqi_timeseries.ipynb          # Full analysis: all 6 stages with statistical narrative
├── AQI_Univariate_TimeSeries.csv # Raw daily data (read-only)
├── outputs/
│   ├── aqi_weekly.csv            # Cleaned weekly series (432 weeks, imputed flag)
│   └── forecast_weekly_2025-04-06_2025-12-28.csv  # Final forecast with 80/95% PI
├── plots/
│   ├── 01_weekly_series_with_imputed.png
│   ├── 02_annual_variance_stability.png
│   ├── 03_level_vs_seasonal_diff.png
│   ├── 04_full_time_series_with_horizon.png
│   ├── 05_stl_decomposition.png
│   ├── 06_seasonal_views.png
│   ├── 07_cpcb_band_distribution.png
│   ├── 08_acf_level.png
│   ├── 09_acf_pacf_D1.png
│   ├── 10_residual_diagnostics.png
│   ├── 11_validation_forecast_comparison.png
│   └── 12_final_forecast.png
├── requirements.txt
├── CLAUDE.md                     # Project specification and development log
└── README.md
```

---

## Forecast Output Format

`outputs/forecast_weekly_2025-04-06_2025-12-28.csv`

| Column | Description |
|---|---|
| `week_ending` | Sunday date of the forecast week |
| `forecast` | Point forecast (AQI) |
| `lower_80` / `upper_80` | 80% prediction interval bounds |
| `lower_95` / `upper_95` | 95% prediction interval bounds |

---

## Setup & Usage

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Run the notebook**
```bash
jupyter notebook aqi_timeseries.ipynb
```

Run cells top to bottom. Each stage opens with a markdown cell stating the statistical purpose, the diagnostic evidence, and the decision taken.

**Requirements:** Python 3.10+, pandas, numpy, statsmodels, scipy, matplotlib, seaborn, jupyter

---

## Limitations

1. **Univariate only** — no meteorological covariates (rainfall, temperature, wind). Adding these via SARIMAX would likely reduce forecast uncertainty for monsoon timing and winter severity.
2. **Linear Gaussian model** — SARIMA does not capture non-linear pollution events (e.g., Diwali fireworks, unseasonal dust storms).
3. **SMA boundary** — the seasonal MA coefficient (≈ −0.97) sits near the invertibility boundary, indicating the model is close to over-differenced at D=1. A future robustness check could explore D=0 with higher-order seasonal ARMA.
4. **Interval calibration** — 80%/95% PIs are theoretically calibrated for this model class; real-world coverage may differ if model assumptions are violated.
