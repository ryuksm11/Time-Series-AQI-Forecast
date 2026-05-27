# AQI Time Series Forecasting

End-to-end Air Quality Index (AQI) forecasting at two temporal resolutions — **weekly** and **daily** — using classical statistical time series methods. Every modelling decision is backed by a named diagnostic or test; no silent steps.

---

## Problem Statement

Forecast AQI for an Indian urban monitoring site across a multi-month horizon (April–December 2025) with point forecasts and 80 %/95 % prediction intervals. The data follows the India CPCB AQI scale (0–500), dominated by a strong annual cycle: monsoon rainfall suppresses AQI in July–September while winter thermal inversions push it into the Very Poor range in November–December.

---

## Data

| Property | Value |
|---|---|
| File | `AQI_Univariate_TimeSeries.csv` |
| Raw frequency | Daily, 2017-01-01 → 2025-12-31 (3,287 rows) |
| Observed span | 2017-01-01 → 2025-03-31 |
| Forecast horizon | 2025-04-01 → 2025-12-31 (275 daily rows, NaN in CSV) |
| Interior missing values | 33 daily NaNs (1.10 % of observed span) |
| AQI range | 41 – 494 |
| Scale | India CPCB AQI |

**CPCB AQI bands:** Good (0–50) · Satisfactory (51–100) · Moderate (101–200) · Poor (201–300) · Very Poor (301–400) · Severe (401–500)

---

## Two Notebooks

| | Weekly (`aqi_weekly_timeseries.ipynb`) | Daily (`aqi_daily.ipynb`) |
|---|---|---|
| Granularity | Weekly mean, Sunday-end | Raw daily |
| Seasonal period | s = 52 | s = 365 (via STL) |
| Model | SARIMA(2,0,0)(0,1,1)₅₂ | STL + ARMA(2,0,1) |
| Forecast horizon | 38 weeks (Apr–Dec 2025) | 275 days (Apr–Dec 2025) |
| Validation MAPE | **16.99 %** | 27.79 % |
| Validation R² | **0.828** | 0.644 |

The weekly notebook aggregates daily data to weekly means (s = 52) so that SARIMA's state-space machinery stays tractable. The daily notebook works at full resolution using STL decomposition, which bypasses the s = 365 intractability.

---

## Weekly Analysis — `aqi_timeseries.ipynb`

### Pipeline

```
Raw daily CSV
      │
      ▼
Stage 1 — Aggregation to weekly + imputation
  • Daily → weekly mean, resample('W'), week-ending Sunday
  • Coverage threshold: ≥ 4 observed days per week
  • 5 weeks imputed (1.2 %): deseasonalise → linear interpolate → reseasonalise
  • Boundary weeks: pure week-of-year climatology
      │
      ▼
Stage 2 — Stationarity & transformation
  • Levene p = 0.960 → homoskedastic → no Box-Cox
  • ADF p ≈ 0 (lags = 12) + KPSS p ≥ 0.10 → d = 0
  • STL Fs = 0.875 → strong seasonality; D = 1 reduces variance 62.5 % → D = 1, s = 52
      │
      ▼
Stage 3 — EDA & model identification
  • STL decomposition (additive, period = 52, robust): Fs = 0.875, Ft = 0.113
  • ACF/PACF of D=1 series: PACF cuts off at lag 2 → AR(2);
    isolated ACF spike at lag 52 (−0.44) → SMA(1)
  • Candidates: SARIMA(1,0,0)(0,1,1)₅₂, (2,0,0)(0,1,1)₅₂,
                (1,0,0)(1,1,0)₅₂, (2,0,0)(1,1,0)₅₂
      │
      ▼
Stage 4 — Modelling & evaluation
  • Train: 380 weeks (2017-01-01 → 2024-04-07)
  • Validation: 52 weeks (2024-04-14 → 2025-04-06)
  • Rolling-origin: 4 origins × 12 steps
  • All 4 SARIMA candidates fitted; refined by AIC/BIC within the diagnostic set
  • Residual diagnostics: Ljung-Box (lags 10, 20, 52), Q-Q, Shapiro-Wilk
      │
      ▼
Stage 5 — Final forecast
  • Refit on full 432-week series
  • 38-week horizon with 80 % and 95 % prediction intervals
```

### Key statistical decisions

| Decision | Value | Justification |
|---|---|---|
| Aggregation | Weekly mean, Sunday-end | s = 52 tractable for SARIMA; s = 365 is not |
| Partial-week threshold | ≥ 4 observed days | Variance ≤ 1.75× full-week variance |
| Variance transform | None | Levene p = 0.960 — homoskedastic |
| d | 0 | ADF p ≈ 0 (lags = 12); KPSS p ≥ 0.10 |
| D | 1, s = 52 | Fs = 0.875; 62.5 % variance reduction; ADF + KPSS confirm post-D=1 stationarity |
| Decomposition | Additive | Homoskedastic variance; consistent seasonal amplitude |
| SARIMA orders | (2,0,0)(0,1,1)₅₂ | PACF cutoff lag 2 → AR(2); ACF spike lag 52 → SMA(1) |
| ETS disqualified | ETS(A,Ad,A) | Ljung-Box fails at lags 20 and 52 |

### Validation results — 52 weeks (2024-04-14 → 2025-04-06)

| Model | MAE | RMSE | MAPE | MASE | R² |
|---|---|---|---|---|---|
| Naive | 81.48 | 102.98 | 44.1 % | 1.702 | −0.21 |
| Drift | 89.12 | 111.93 | 45.4 % | 1.861 | −0.42 |
| Seasonal Naive (lag 52) | 39.69 | 49.17 | 22.3 % | 0.829 | 0.725 |
| ETS(A,Ad,A) | 31.44 | 39.99 | 16.7 % | 0.657 | 0.818 |
| **SARIMA(2,0,0)(0,1,1)₅₂** | **30.68** | **38.95** | **17.0 %** | **0.641** | **0.828** |

SARIMA wins on every metric and passes Ljung-Box at all lags including lag 52 (p = 0.730).

### Final forecast (38 weeks, 2025-04-13 → 2025-12-28)

| Month | Mean AQI | CPCB Band |
|---|---|---|
| Apr–May | ~197–211 | Moderate |
| Jun–Sep | 92–158 | Satisfactory / Moderate |
| Oct | ~248 | Poor |
| Nov–Dec | 335–346 | Very Poor |

---

## Daily Analysis — `aqi_daily.ipynb`

### Pipeline

```
Raw daily CSV
      │
      ▼
Stage 1 — Data load & imputation
  • Validate: 3,287 rows, 0 date gaps, 33 interior NaNs, 275 trailing NaNs
  • Impute 33 NaNs: time-based linear interpolation
    (all gaps within one seasonal regime — no seasonal correction needed)
      │
      ▼
Stage 2 — Stationarity & transformation
  • Levene p = 0.484 → homoskedastic → no Box-Cox
  • ADF p = 0.000137 (lags = 29) + KPSS p ≥ 0.10 → d = 0
  • Seasonal differencing not applicable: annual cycle handled by STL
      │
      ▼
Stage 3 — EDA & model identification
  • STL decomposition (additive, period = 365, robust): Fs = 0.769, Ft = 0.057
  • Fourier-ARIMA (K = 1..10): disqualified — all variants fail Ljung-Box,
    MAPE ≈ 100 % (high AR persistence overwhelms Fourier shape at 275-day horizon)
  • STL remainder PACF cuts off at lag 1 → AR(1) base;
    ARMA(2,0,1) selected by AIC, confirmed by Ljung-Box
      │
      ▼
Stage 4 — Modelling & evaluation
  • Train: 2,647 days (2017-01-01 → 2024-03-31)
  • Validation: 365 days (2024-04-01 → 2025-03-31)
  • Rolling-origin: 4 origins × 30 steps
  • Residual diagnostics: Ljung-Box (lags 10, 20, 30), Q-Q, Shapiro-Wilk
      │
      ▼
Stage 5 — Final forecast
  • Refit on full 3,012-day observed series
  • 275-day horizon with 80 % and 95 % prediction intervals
```

### Key statistical decisions

| Decision | Value | Justification |
|---|---|---|
| Imputation | Time-based linear interpolation | All 33 gaps within one seasonal regime; 23-day Aug gap stays in monsoon band |
| Variance transform | None | Levene p = 0.484 — homoskedastic |
| d | 0 | ADF p = 0.000137 (lags = 29); KPSS p ≥ 0.10 |
| D | N/A | Annual cycle extracted by STL; no seasonal differencing required |
| Decomposition | Additive STL, period = 365 | Fs = 0.769; homoskedastic variance |
| Fourier-ARIMA | Disqualified | All K = 1..10 fail Ljung-Box and produce MAPE ≈ 100 % |
| ARMA order | (2,0,1) on STL remainder | Lowest AIC; passes Ljung-Box at lags 10, 20, 30 |

### Validation results — 365 days (2024-04-01 → 2025-03-31)

| Model | MAE | RMSE | MAPE | MASE | R² |
|---|---|---|---|---|---|
| Naive | 92.35 | 108.12 | 74.5 % | 1.428 | −0.15 |
| Drift | 91.15 | 107.12 | 72.1 % | 1.410 | −0.13 |
| Seasonal Naive (lag 365) | 55.21 | 71.57 | 32.6 % | 0.854 | 0.497 |
| **STL + ARMA(2,0,1)** | **45.55** | **60.17** | **27.8 %** | **0.704** | **0.644** |

### Final forecast (275 days, 2025-04-01 → 2025-12-31)

| Month | Mean AQI | CPCB Band |
|---|---|---|
| Apr–May | ~187 | Moderate |
| Jun–Sep | 74–139 | Satisfactory / Moderate |
| Oct | ~230 | Poor |
| Nov–Dec | 313–342 | Very Poor |

---

## Project Structure

```
Time_Series_AQI_Forecast/
├── aqi_weekly_timeseries.ipynb                   # Weekly analysis (Stages 1–5)
├── aqi_daily.ipynb                               # Daily analysis (Stages 1–5)
├── AQI_Univariate_TimeSeries.csv                 # Raw daily data (read-only)
├── outputs/
│   ├── aqi_weekly.csv                            # Cleaned weekly series (432 weeks)
│   ├── forecast_weekly_2025-04-06_2025-12-28.csv # Weekly forecast + 80/95 % PI
│   └── forecast_daily_2025-04-01_2025-12-31.csv  # Daily forecast + 80/95 % PI
├── plots/                                        # Weekly analysis figures (12 plots)
├── plots_daily/                                  # Daily analysis figures (11 plots)
├── requirements.txt
└── README.md
```

### Forecast output columns

| Column | Description |
|---|---|
| `week_ending` / `date` | Sunday date (weekly) or calendar date (daily) |
| `forecast` | Point forecast (AQI) |
| `lower_80` / `upper_80` | 80 % prediction interval |
| `lower_95` / `upper_95` | 95 % prediction interval |

---

## Setup & Usage

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Run the weekly notebook**
```bash
jupyter notebook aqi_weekly_timeseries.ipynb
```

**Run the daily notebook**
```bash
jupyter notebook aqi_daily.ipynb
```

Run cells top to bottom in either notebook. Each stage opens with a markdown cell stating the statistical purpose, the diagnostic evidence, and the decision taken.

**Requirements:** Python 3.10+, pandas, numpy, statsmodels, scipy, matplotlib, seaborn, jupyter

---

## Limitations

1. **Univariate only** — no meteorological covariates (rainfall, temperature, wind). Adding these via SARIMAX would tighten intervals, especially for monsoon onset/withdrawal timing.
2. **Fourier-ARIMA failure (daily)** — the asymmetric AQI seasonal pattern and high AR persistence make Fourier-ARIMA unsuitable at daily resolution. STL decomposition is the correct approach.
3. **Linear Gaussian models** — neither SARIMA nor ARMA captures non-linear pollution events (Diwali firecrackers, unseasonal dust storms).
4. **Daily interval width** — the 95 % PI stabilises at ≈ ±99 AQI after day 30, reflecting irreducible noise at daily resolution without covariates.
