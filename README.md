# Bayesian Gaussian Process Regression for Missing Sleep Latency Data



This repository contains the implementation of a Bayesian Gaussian Process (GP) regression framework for imputing missing sleep latency outcomes in longitudinal clinical trials, as described in the paper:

> Guo, C. (2025). Bayesian Gaussian Process Regression for Imputing Missing Sleep Latency Outcomes in Longitudinal Clinical Trials. *PHP2530*.

## Overview

Sleep disorders affect millions worldwide, with insomnia being among the most prevalent. Clinical trials investigating insomnia treatments commonly measure sleep latency (time to fall asleep) as a primary endpoint, but often encounter substantial missing data that can bias results. This project implements a sparse additive Gaussian Process regression framework specifically designed for imputing missing sleep latency outcomes.

The approach:
- Flexibly models nonlinear temporal dynamics and complex covariate interactions
- Provides coherent uncertainty quantification through Bayesian inference
- Uses sparse approximations to make GP models computationally tractable

## Repository Contents

- `Final_code.R`: Main R script containing data simulation, exploratory data analysis (EDA), Multiple Imputation by Chained Equations (MICE) implementations, and performance evaluation
- `sparse_gp.stan`: Stan model implementation of the sparse Gaussian Process for missing data imputation

## Required Packages

```r
# Core packages
library(truncnorm)
library(ggplot2)
library(dplyr)
library(ggthemes)
library(mice)
library(miceadds)
library(rstan)
library(RColorBrewer)
library(bayesplot)
```

## Implementation Details

The implementation follows these key steps:

1. **Data Simulation**: Simulates a longitudinal clinical trial with 66 participants across three treatment arms (placebo, Drug A, Drug B) tracked for 21 days, with nonlinear treatment effects and realistic covariates

2. **Missing Data Generation**: Creates a mixed Missing Completely at Random (MCAR) and Missing at Random (MAR) pattern with approximately 30% of data missing

3. **Multiple Imputation Methods**:
   - Predictive Mean Matching (PMM)
   - Random Forests (RF)
   - Normal Linear Regression (norm)

4. **Sparse Gaussian Process Implementation**:
   - Employs inducing points for computational efficiency
   - Uses Hamiltonian Monte Carlo via Stan for posterior sampling
   - Computes posterior predictive distributions for missing values

5. **Performance Evaluation**: Compares imputation methods based on bias, RMSE, coverage, and credible interval width

## Usage

### Simulating Data and Baseline Models

To run the data simulation and MICE imputation methods:

```r
# Run the R script
source("Final_code.R")
```

### Running the Sparse GP Model

To fit the Stan model:

```r
# Compile the Stan model
gp_mod <- stan_model(file = "sparse_gp.stan")

# Run the MCMC sampling
fit_gp <- sampling(gp_mod, data = stan_data,
                   iter = 2000, warmup = 1000,
                   chains = 4, cores = 4, refresh = 50,
                   control = list(adapt_delta = 0.95))
```

## Key Results

Comparative analysis showed:

1. The Sparse GP approach demonstrated superior performance in terms of root mean squared error (RMSE = 5.35) compared to MICE variants (5.95 to 6.27)

2. Most significantly, the GP method achieved approximately 90% coverage probability compared to only 61-64% for MICE methods

3. The improved coverage came with wider credible intervals, exemplifying the classical bias-variance tradeoff

## Visualization

The repository includes code for visualizing:

- Missing data patterns over time
- Treatment-specific temporal trajectories
- Posterior distributions of model parameters
- Performance comparisons across imputation methods

