# AQI Daily Time Series Forecasting

Daily Air Quality Index (AQI) forecasting using STL decomposition combined with ARMA modelling. Every modelling decision is backed by a named statistical diagnostic or test.

---

## Problem Statement

Forecast daily AQI for an Indian urban monitoring site across a 275-day horizon (2025-04-01 → 2025-12-31) with point forecasts and 80 %/95 % prediction intervals. The data follows the India CPCB AQI scale (0–500), driven by a strong annual cycle — monsoon rainfall suppresses AQI in July–September while winter thermal inversions push it into the Very Poor range in November–December.

---

## Data

| Property | Value |
|---|---|
| File | `AQI_Univariate_TimeSeries.csv` |
| Raw frequency | Daily, 2017-01-01 → 2025-12-31 (3,287 rows) |
| Observed span | 2017-01-01 → 2025-03-31 (3,012 days) |
| Forecast horizon | 2025-04-01 → 2025-12-31 (275 days, NaN in CSV) |
| Interior missing values | 33 daily NaNs (genuine gaps, 1.10 %) |
| AQI range | 41 – 494 (post-imputation) |
| Scale | India CPCB AQI |

**CPCB AQI bands:** Good (0–50) · Satisfactory (51–100) · Moderate (101–200) · Poor (201–300) · Very Poor (301–400) · Severe (401–500)

---

## Methodology Pipeline

```
Raw daily CSV (3,287 rows)
        │
        ▼
Stage 1 — Data load & imputation
  • Validate: 3,287 rows, 0 date gaps, 33 interior NaNs, 275 trailing NaNs
  • Impute 33 interior NaNs: time-based linear interpolation
    - Eight 1-day gaps, one 3-day gap, one 23-day gap (Aug 2017), one leap-day gap
    - All gaps lie within a single seasonal regime → linear fill is appropriate
        │
        ▼
Stage 2 — Stationarity & transformation
  • Variance: Levene p = 0.484 → homoskedastic → no Box-Cox transform
  • Level stationarity: ADF p = 0.000137 (lags = 29) + KPSS p ≥ 0.10 → d = 0
  • Seasonal differencing: N/A — annual cycle handled by STL decomposition
        │
        ▼
Stage 3 — EDA & model identification
  • STL decomposition (additive, period = 365, robust)
    - Seasonal strength Fs = 0.769 >> 0.64 → annual cycle dominates
    - Trend strength Ft = 0.057 → negligible long-run trend
  • Fourier-ARIMA evaluated (K = 1 to 10): disqualified
    - All variants fail Ljung-Box and produce MAPE ≈ 100 %
    - High AR persistence (φ₁ ≈ 0.87) dominates the 275-day horizon
  • STL remainder ACF/PACF: PACF cuts off at lag 1 → AR(1) base;
    ARMA(2,0,1) selected by AIC and confirmed by Ljung-Box
        │
        ▼
Stage 4 — Modelling & evaluation
  • Train: 2,647 days (2017-01-01 → 2024-03-31)
  • Validation: 365 days (2024-04-01 → 2025-03-31)
  • Rolling-origin backtesting: 4 origins × 30 steps
  • Models: Naive, Seasonal Naive (lag 365), Drift, STL+ARMA(2,0,1)
  • Residual diagnostics: Ljung-Box (lags 10, 20, 30), Q-Q, Shapiro-Wilk
        │
        ▼
Stage 5 — Final forecast
  • Refit on full 3,012-day observed series
  • 275-day forecast with 80 % and 95 % prediction intervals
  • Output: CSV + forecast plot with CPCB band overlay
```

---

## Key Statistical Decisions

| Decision | Value | Justification |
|---|---|---|
| Imputation | Time-based linear interpolation | All 33 gaps within one seasonal regime; no regime boundary crossed |
| Variance transform | None | Levene p = 0.484 — homoskedastic across all years |
| Regular differencing | d = 0 | ADF p = 0.000137 (lags = 29); KPSS stat = 0.058, p ≥ 0.10 |
| Seasonal differencing | N/A | Annual cycle extracted by STL; no D required |
| Decomposition | Additive STL, period = 365 | Homoskedastic variance; Fs = 0.769 confirms dominant annual cycle |
| Fourier-ARIMA | Disqualified | All K = 1..10 fail Ljung-Box; MAPE ≈ 100 % at all K |
| ARMA order | (2, 0, 1) on STL remainder | Lowest AIC; passes Ljung-Box at lags 10, 20, 30 |

---

## Results

### Validation set (365 days, 2024-04-01 → 2025-03-31)

| Model | MAE | RMSE | MAPE | MASE | R² |
|---|---|---|---|---|---|
| Naive | 92.35 | 108.12 | 74.5 % | 1.428 | −0.15 |
| Drift | 91.15 | 107.12 | 72.1 % | 1.410 | −0.13 |
| Seasonal Naive (lag 365) | 55.21 | 71.57 | 32.6 % | 0.854 | 0.497 |
| **STL + ARMA(2,0,1)** | **45.55** | **60.17** | **27.8 %** | **0.704** | **0.644** |

STL + ARMA(2,0,1) is **18 % more accurate than Seasonal Naive** (MASE 0.704 vs 0.854) and explains **64 % of daily variance**. Rolling-origin backtesting across 4 origins confirms the ranking is stable (MASE 0.555 vs 0.856).

### Final forecast (275 days, 2025-04-01 → 2025-12-31)

| Month | Mean AQI | CPCB Band | Driver |
|---|---|---|---|
| Apr – May | ~187 | Moderate | Post-winter transition |
| Jun – Sep | 74 – 139 | Satisfactory / Moderate | Monsoon rainfall |
| Oct | ~230 | Poor | Post-monsoon drying |
| Nov – Dec | 313 – 342 | Very Poor | Winter inversion + stubble burning |

95 % prediction interval stabilises at ≈ ±99 AQI by day 30 (AR memory decays; uncertainty driven by seasonal innovation variance alone for the remaining 245 days).

---

## Project Structure

```
Time_Series_AQI_Forecast/
├── aqi_daily.ipynb                              # Full analysis: all 5 stages
├── AQI_Univariate_TimeSeries.csv                # Raw daily data (read-only)
├── outputs/
│   └── forecast_daily_2025-04-01_2025-12-31.csv
├── plots_daily/
│   ├── 01_daily_series_with_imputed.png
│   ├── 02_annual_variance_stability.png
│   ├── 03_level_vs_diff.png
│   ├── 04_stl_decomposition.png
│   ├── 05_seasonal_views.png
│   ├── 06_acf_pacf_level.png
│   ├── 07_stl_remainder_acf_pacf.png
│   ├── 08_residual_diagnostics.png
│   ├── 09_validation_forecast.png
│   └── 10_final_forecast.png
├── requirements.txt
└── README.md
```

---

## Forecast Output Format

`outputs/forecast_daily_2025-04-01_2025-12-31.csv`

| Column | Description |
|---|---|
| `date` | Calendar date of the forecast day |
| `forecast` | Point forecast (AQI) |
| `lower_80` / `upper_80` | 80 % prediction interval bounds |
| `lower_95` / `upper_95` | 95 % prediction interval bounds |

---

## Setup & Usage

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Run the notebook**
```bash
jupyter notebook aqi_daily.ipynb
```

Run cells top to bottom. Each stage opens with a markdown cell stating the statistical purpose, the diagnostic evidence, and the decision taken.

**Requirements:** Python 3.10+, pandas, numpy, statsmodels, scipy, matplotlib, seaborn, jupyter

---

## Limitations

1. **Fourier-ARIMA failure** — the asymmetric Indian AQI seasonal pattern (sharp monsoon trough, gradual winter build-up) cannot be captured by Fourier harmonics when the AR persistence is high. STL decomposition is the correct approach for this dataset.
2. **No meteorological covariates** — rainfall, temperature, and wind speed drive monsoon onset/withdrawal timing and winter inversion severity. Adding these via SARIMAX would substantially tighten prediction intervals.
3. **Wide daily intervals** — the 95 % PI of ≈ ±99 AQI after day 30 reflects genuine irreducible noise at daily resolution without covariates.
4. **Linear Gaussian model** — ARMA does not capture non-linear pollution events such as Diwali firecrackers or unseasonal dust storms.
