# Bayesian Multi-Site Air Quality Prediction

This repository contains code and Jupyter notebooks for hierarchical, pooled, and quantized Bayesian modeling of air quality time series data from multiple monitoring stations in Beijing and Santiago. The models leverage JAX, Flax, and SGMCMC methods for scalable Bayesian inference, and include quantization and Stein thinning for efficient posterior sample storage and diagnostics.

## Repository Structure

```
.
├── beijing_multi_site_hierarchical.ipynb   # Hierarchical modeling for Beijing data
├── beijing_multi_site.ipynb                # Multi-site modeling for Beijing data
├── santiago_multi_site.ipynb               # Multi-site modeling for Santiago data
├── hierarchical_model.py                   # Python script for hierarchical model
├── pooled_model.py                         # Python script for pooled model
├── normal_model.py                         # Python script for normal model
├── lstm_jax.py                             # LSTM model in JAX/Flax
├── quantized_mixture_model.ipynb           # Quantized mixture model experiments
├── quantized_sgmcmc_diagnostics.ipynb      # Diagnostics for quantized SGMCMC
├── run_experiments.sh                      # Shell script to run experiments
├── data/                                   # Directory for preprocessed and posterior data
├── data_completa11/                        # Directory for raw Santiago CSVs
├── stein_thinning/                         # Stein thinning utilities
├── docs/                                   # Documentation
├── figures/                                # Generated figures
└── .vscode/                                # VSCode settings
```

## Main Features

- **Data Preprocessing:** Scripts to load, clean, and preprocess air quality data from CSV files.
- **Hierarchical & Pooled Models:** Bayesian LSTM models for multi-site time series forecasting.
- **SGMCMC Inference:** Stochastic gradient MCMC for scalable Bayesian inference.
- **Posterior Diagnostics:** Effective sample size, Kernel Stein Discrepancy, and Stein thinning.
- **Quantization:** Posterior sample quantization (float16, int8) for efficient storage.
- **Visualization:** Notebooks for plotting predictions, uncertainty, and diagnostics.

## Getting Started

### Requirements

- Python 3.8+
- JAX, Flax, Optax, Distrax
- NumPy, Pandas, Matplotlib, Scikit-learn
- stein_thinning (custom module)

Install dependencies (example using pip):

```sh
pip install jax flax optax distrax numpy pandas matplotlib scikit-learn
```

### Data

- Place Beijing CSVs in PRSA_Data_20130301-20170228
- Place Santiago CSVs in data_completa11

### Running Experiments

You can run the main experiments and diagnostics by opening and executing the notebooks:

- beijing_multi_site.ipynb
- santiago_multi_site.ipynb
- quantized_mixture_model.ipynb

Or run the provided shell script:

```sh
bash run_experiments.sh
```

## Notebooks Overview

- **beijing_multi_site.ipynb:** End-to-end pipeline for Beijing data, including preprocessing, model training, SGMCMC sampling, diagnostics, and visualization.
- **santiago_multi_site.ipynb:** Same as above, for Santiago data.
- **quantized_mixture_model.ipynb:** Experiments with quantized posterior samples and mixture models.

## Posterior Diagnostics

- **Effective Sample Size:** Computed using `effective_sample_size`.
- **Kernel Stein Discrepancy:** Computed using `kernel_stein_discrepancy_imq`.
- **Stein Thinning:** See stein_thinning and usage in notebooks.

## Citation

If you use this code, please cite the corresponding paper or this repository.

## License

MIT License

---

For questions or contributions, please open an issue or pull request.