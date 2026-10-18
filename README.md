# AI-Powered Sales Forecasting Platform

An end-to-end, locally deployable sales forecasting application that
transforms point-of-sale (POS) data into short-term revenue forecasts,
product-demand insights, and interactive business analytics.

> **Project context:** Engineering final-year project (PFE) developed in
> the context of Twins Digital Labs / SCICITE Twins Lab, using BPrice
> POS data.
>
> **Note:** This README describes the project architecture and reported
> experiments. Update the setup commands, filenames, screenshots, and
> configuration details to match the current repository before
> publishing.

## Table of Contents

-   [Overview](#overview)
-   [Key Features](#key-features)
-   [Architecture](#architecture)
-   [Machine Learning](#machine-learning)
-   [Reported Results](#reported-results)
-   [Technology Stack](#technology-stack)
-   [Project Structure](#project-structure)
-   [Getting Started](#getting-started)
-   [Data Format](#data-format)
-   [Usage](#usage)
-   [Evaluation and Limitations](#evaluation-and-limitations)
-   [Roadmap](#roadmap)
-   [Author](#author)
-   [License](#license)

## Overview

Retail businesses generate large amounts of POS data, but turning
transaction history into reliable, actionable forecasts can be
difficult. This project provides a practical pipeline for preparing
sales data, training and evaluating forecasting models, and presenting
the results through a Streamlit dashboard.

The application is designed for **local, single-machine deployment** and
does not require a cloud-hosted inference service for its core workflow.

## Key Features

-   **POS data ingestion:** Supports JSON and CSV input formats.
-   **Data cleaning:** Standardizes fields, handles missing values, and
    prepares consistent records for analysis.
-   **Fuzzy field mapping:** Helps map differently named source columns
    to the expected schema.
-   **Daily sales aggregation:** Converts transaction-level records into
    daily time-series data.
-   **Missing-date handling:** Supports filling gaps in daily time
    series before model training.
-   **Weather enrichment:** Integrates weather variables from NASA POWER
    when configured and available.
-   **Feature engineering:** Uses temporal, calendar, lag, and
    rolling-window features.
-   **Model comparison:** Evaluates XGBoost, LightGBM, CatBoost, Random
    Forest, and LSTM.
-   **Revenue forecasting:** Generates recursive forecasts for the next
    seven days.
-   **Product-demand insights:** Produces a ranked list of up to 20
    products by predicted demand, where supported by the available data.
-   **Prediction intervals:** Presents 95% intervals when the configured
    interval-generation method is available. These should not be
    interpreted as guaranteed coverage.
-   **Interactive dashboard:** Displays forecasts, evaluation metrics,
    and business-oriented visualizations in Streamlit.
-   **Local AI assistant:** The project design includes a
    retrieval-based assistant using TF-IDF and an LLM to answer
    questions grounded in available sales information.

## Architecture

``` text
POS data (CSV / JSON)
        |
        v
Schema mapping and validation
        |
        v
Cleaning and preprocessing
        |
        v
Daily aggregation and missing-date handling
        |
        +----------------------+
        |                      |
        v                      v
Feature engineering      Weather enrichment
        |                 (NASA POWER, optional)
        +----------+-----------+
                   |
                   v
        Chronological train/test split
                   |
                   v
       Train and evaluate forecasting models
                   |
                   v
       Select/compare the best-performing models
                   |
          +--------+---------+
          |                  |
          v                  v
  7-day revenue forecast   Product-demand ranking
          |                  |
          +--------+---------+
                   |
                   v
        Streamlit analytics dashboard
                   |
                   v
       Optional local retrieval-based assistant
```

## Machine Learning

The forecasting workflow compares multiple model families rather than
assuming that one algorithm is best for every dataset.

  -----------------------------------------------------------------------
  Model                               Role
  ----------------------------------- -----------------------------------
  XGBoost                             Gradient-boosted trees for
                                      engineered time-series features

  LightGBM                            Efficient gradient boosting for
                                      structured data

  CatBoost                            Gradient boosting with robust
                                      handling of categorical features
                                      when configured

  Random Forest                       Tree-ensemble baseline

  LSTM                                Recurrent neural network baseline
                                      for sequential patterns
  -----------------------------------------------------------------------

### Forecasting workflow

1.  Validate and normalize the input data.
2.  Aggregate sales into a daily time series.
3.  Create temporal, calendar, lag, rolling, and optional weather
    features.
4.  Split the data chronologically to reduce time-series leakage.
5.  Train candidate models and evaluate them on held-out data.
6.  Compare metrics and select the model according to the evaluation
    strategy.
7.  Generate recursive forecasts for the next seven days.
8.  Display forecast values and available uncertainty intervals in the
    dashboard.

**Important:** Recursive forecasting feeds earlier predictions into
later forecast steps. Errors can accumulate across the forecast horizon,
so results should be checked against suitable baselines and refreshed as
new actual sales arrive.

## Reported Results

The following are the project's previously reported XGBoost evaluation
results. They are included for documentation and have **not been
independently re-run as part of creating this README**.

  Metric     Reported value
  -------- ----------------
  MAE                 81.96
  RMSE               103.69
  MAPE                8.83%
  SMAPE              23.49%
  R²                   0.83

The project also compared LightGBM, CatBoost, Random Forest, and LSTM.
In the reported experiment, XGBoost was the leading model overall.

### Interpreting the metrics

-   **MAE:** Average absolute prediction error, expressed in the
    target's units.
-   **RMSE:** Gives larger errors more weight than MAE.
-   **MAPE:** Percentage-based error; it can be unstable when actual
    values are zero or near zero.
-   **SMAPE:** Symmetric percentage-based error measure.
-   **R²:** Measures how much target variance is explained relative to a
    mean-prediction baseline.

The MAE and RMSE units depend on how the revenue target is represented
in the source data. Performance can vary by store, product mix, season,
forecast horizon, and data quality. Always evaluate against a
chronological holdout and a simple baseline.

## Technology Stack

The project uses the following tools and techniques:

-   **Language:** Python
-   **Dashboard:** Streamlit
-   **Machine learning:** XGBoost, LightGBM, CatBoost, Random Forest,
    and LSTM
-   **Data processing:** Pandas and NumPy
-   **Evaluation:** MAE, RMSE, MAPE, SMAPE, and R²
-   **Data enrichment:** NASA POWER weather data
    (optional/configuration-dependent)
-   **Assistant component:** TF-IDF retrieval and an LLM (local
    configuration-dependent)
-   **Deployment approach:** Local, single-machine application

Check the repository's dependency file for exact package versions.

## Project Structure

The exact structure depends on the current repository. A possible
organization is:

``` text
sales-forecasting/
├── app.py                    # Streamlit application entry point
├── data/
│   ├── raw/                  # Input POS data (not committed if private)
│   └── processed/             # Prepared datasets (often generated)
├── models/                   # Saved model artifacts (if used)
├── src/
│   ├── data_processing.py    # Cleaning and schema mapping
│   ├── feature_engineering.py
│   ├── train.py              # Model training
│   ├── evaluate.py            # Evaluation utilities
│   └── forecasting.py        # Forecast generation
├── tests/                    # Tests (if available)
├── requirements.txt          # Python dependencies
├── .env.example              # Optional configuration template
└── README.md
```

This is an illustrative structure, not a claim about the exact files in
your repository. Replace it with the real project tree before
publishing.

## Getting Started

### 1. Prerequisites

-   Python version supported by the project's dependencies.
-   Git, if cloning the repository.
-   A CSV or JSON export of POS data in the schema expected by the
    application.
-   Internet access only for optional external data retrieval, such as
    NASA POWER, and for installing dependencies.

### 2. Get the source code

If the project is hosted in a Git repository:

``` bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_PROJECT_DIRECTORY>
```

Replace the placeholders with the repository URL and directory.

### 3. Create and activate a virtual environment

**Windows (PowerShell):**

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install dependencies

If the repository includes `requirements.txt`:

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If it uses another dependency manager, follow the repository's lockfile
and instructions instead.

### 5. Configure the application

Review the project's configuration files and set any required paths,
model locations, or API parameters. If weather enrichment or the local
LLM requires additional configuration, follow the instructions in the
relevant module.

Do not commit API keys, credentials, private POS exports, or customer
information. Use environment variables or a local `.env` file if the
application supports them.

### 6. Run the dashboard

If `app.py` is the actual Streamlit entry point, start it with:

``` bash
streamlit run app.py
```

If the entry point has a different name or path, substitute the correct
file. Streamlit will print a local URL in the terminal.

## Data Format

The ingestion pipeline was designed to accept POS exports in CSV or
JSON, with schema mapping used to accommodate differences in field
names. The exact required columns depend on the implementation.

A typical daily sales dataset may contain fields such as:

  -----------------------------------------------------------------------
  Field                               Description
  ----------------------------------- -----------------------------------
  `date`                              Transaction date or business date

  `product_id`                        Product identifier, when
                                      product-level forecasting is
                                      required

  `quantity`                          Units sold

  `revenue`                           Sales amount

  `store_id`                          Store identifier, if multiple
                                      stores are supported
  -----------------------------------------------------------------------

These are **illustrative field names**, not guaranteed input
requirements. Check the data-processing code for the actual schema and
accepted aliases.

Before using real business data, remove or protect personal information
and verify that you have permission to process the data. Do not publish
raw POS records in a public repository.

## Usage

A typical workflow is:

1.  Launch the Streamlit application.
2.  Upload or select the POS dataset.
3.  Run validation and preprocessing.
4.  Review the cleaned data and any detected schema mappings.
5.  Train or load the forecasting models, according to the application's
    supported workflow.
6.  Compare model metrics.
7.  Generate the seven-day revenue forecast and available product-demand
    rankings.
8.  Inspect the dashboard charts and uncertainty intervals.
9.  Export results if the application provides export controls.
10. Refresh the forecast as new sales data becomes available.

The exact interface labels and available buttons depend on the current
implementation.

## Evaluation and Limitations

-   **Time-series validation matters:** Random train/test splitting can
    leak future information. Use chronological splits or rolling-origin
    evaluation.
-   **Feature leakage must be prevented:** Lag and rolling features must
    use only information available at each forecast origin.
-   **Forecast intervals need validation:** A nominal 95% interval is
    useful only if its coverage and width are assessed on held-out
    periods.
-   **MAPE has edge cases:** Zero or near-zero actual sales can make it
    misleading.
-   **External data can fail:** Weather enrichment depends on data
    availability, location mapping, and successful retrieval.
-   **Product rankings depend on data quality:** Product identifiers and
    quantities must be consistent for meaningful demand forecasts.
-   **Results are dataset-specific:** The reported metrics are
    experimental results, not a guarantee of future accuracy.
-   **Local resource limits apply:** Training time and model size depend
    on the machine, dataset, and selected algorithms.

## Roadmap

Potential improvements include:

-   Add automated tests for preprocessing and feature generation.
-   Add rolling-origin backtesting and seasonal-naive baselines.
-   Track prediction-interval coverage and calibration.
-   Add model versioning and reproducible training configurations.
-   Improve monitoring for missing data and changing sales patterns.
-   Add per-store and per-product evaluation where the data supports it.
-   Document model retraining and forecast refresh procedures.
-   Add screenshots, a short demo video, and a reproducible sample
    dataset.

## Author

**Ali Hmaid**\
AI and Data Science Engineer

Add your preferred contact links before publishing:

-   LinkedIn: `<YOUR_LINKEDIN_URL>`
-   GitHub: `<YOUR_GITHUB_URL>`
-   Email: `<YOUR_PUBLIC_EMAIL>`

## License

No license has been specified here. Add a `LICENSE` file before
distributing the project publicly. If this is company-owned or academic
work governed by an agreement, confirm the applicable permissions before
publishing source code or data.
