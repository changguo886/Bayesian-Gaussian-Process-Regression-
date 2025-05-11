library(truncnorm)
library(ggplot2)
library(dplyr)
library(ggthemes)
library(mice)
library(miceadds)  # for the gam method
library(rstan)
library(RColorBrewer)


set.seed(123)

# Parameters
n <- 66
T_obs <- 21
sigma <- 5
n_per_group <- 22

# Simulate baseline covariates
age <- round(rnorm(n, mean = 50, sd = 10), 2)                 # age in years
gender_block <- rep(c(0, 1), length.out = 22)
gender <- c(gender_block, gender_block, gender_block)  # balanced per group: 0 = male, 1 = female
baseline_sl <- round(rtruncnorm(n, a = 0, b = 60, mean = 30, sd = 10), 2)

# Simulate GAD-7 score and normalize
gad7_score <- sample(0:21, size = n, replace = TRUE, prob = dpois(7, 0:21) / sum(dpois(7, 0:21)))
anxiety_norm <- round(gad7_score / 21, 3)

# imulate skewed latent (unobserved) caffeine intake using Gamma
caffeine <- round(rgamma(n, shape = 4, rate = 0.02), 1)  # mg/day

# Balanced treatment group assignment
treatment <- rep(c(0, 1, 2), each = n_per_group)  # 0 = Placebo, 1 = Drug A, 2 = Drug B

# Expand to long format
id <- rep(1:n, each = T_obs)
time <- rep(1:T_obs, times = n)
age_long <- rep(age, each = T_obs)
gender_long <- rep(gender, each = T_obs)
baseline_long <- rep(baseline_sl, each = T_obs)
gad7_long <- rep(gad7_score, each = T_obs)
anxiety_long <- rep(anxiety_norm, each = T_obs)
caffeine_long <- rep(caffeine, each = T_obs)
treatment_long <- rep(treatment, each = T_obs)

# Time × Treatment interaction effect
time_effect <- numeric(length(time))
for (i in seq_along(time)) {
  t <- time[i]
  tr <- treatment_long[i]
  if (tr == 0) {
    time_effect[i] <- 20 - 0.2 * t
  } else if (tr == 1) {
    time_effect[i] <- 20 * exp(-0.05 * t)
  } else if (tr == 2) {
    time_effect[i] <- 20 - 20 / (1 + exp(-0.1 * (t - 30)))
  }
}

# Nonlinear covariate effects

# Age: nonlinear (gentle curve)
f_age <- 0.1 * age_long + 3 * sin(age_long / 10)

# Gender: categorical, binary shift
f_gender <- ifelse(gender_long == 1, 3, -2)  # Female: +3, Male: -2

# Baseline SL: slightly nonlinear (saturating)
f_baseline <- 0.2 * baseline_long + 5 * log(1 + baseline_long)

# Anxiety: strongly nonlinear (log-shaped)
f_anxiety <- -10 * log(1 + anxiety_long)

# Caffeine (latent): small nonlinear effect
f_caffeine <- 0.01 * caffeine_long + 2 * log(1 + caffeine_long)



# Latent function and observed sleep latency
f <- round(time_effect + f_age + f_gender + f_baseline + f_anxiety + f_caffeine, 1)
sl <- round(f + rnorm(n * T_obs, mean = 0, sd = sigma))  # sleep latency in minutes



# Final dataset — caffeine kept for diagnostics only
sim_data <- data.frame(
  id = id,
  time = time,
  treatment = factor(treatment_long, levels = c(0, 1, 2), labels = c("Placebo", "DrugA", "DrugB")),
  age = age_long,
  gender = gender_long,
  baseline = baseline_long,
  gad7 = gad7_long,
  anxiety = anxiety_long,
  caffeine = caffeine_long,  # keep only for EDA/validation, not in modeling
  f = f,
  sl = sl
)

# Optional preview
head(sim_data)

##===============================EDA======================================
my_theme <- theme_minimal() +
  theme(
    axis.text = element_text(size = 14),
    axis.title = element_text(size = 14, face = "bold"),
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
  )

# 1. Histogram of Sleep Latency (SL)
ggplot(sim_data, aes(x = sl)) +
  geom_histogram(binwidth = 5, fill = "steelblue", color = "white") +
  labs(title = "Distribution of Sleep Latency (SL)", x = "Sleep Latency (min)", y = "Count") +
  my_theme


#  2. Sleep Latency Over Time by Treatment
ggplot(sim_data, aes(x = time, y = sl, group = id, color = treatment)) +
  geom_line(alpha = 0.1) +
  stat_summary(fun = mean, geom = "line", size = 1.2, aes(group = treatment)) +
  labs(title = "Sleep Latency Over Time by Treatment", y = "Sleep Latency (min)", x = "Time") +
  my_theme

# 3. SL vs. Latent Function f
ggplot(sim_data, aes(x = f, y = sl)) +
  geom_point(alpha = 0.2) +
  geom_smooth(method = "lm", color = "darkred") +
  labs(title = "Observed SL vs. Latent Function", x = "Latent f", y = "Sleep Latency (min)") +
  my_theme

# 4. SL vs. Age by Treatment
ggplot(sim_data, aes(x = age, y = sl, color = treatment)) +
geom_point(alpha = 0.2) +
  geom_smooth(method = "lm") +
  labs(title = "Sleep Latency vs. Age by Treatment", x = "Age", y = "SL (min)") +
  my_theme

#  5. Boxplot of SL by Gender
ggplot(sim_data, aes(x = factor(gender, labels = c("Male", "Female")), y = sl, fill = factor(gender))) +
  geom_boxplot() +
  labs(title = "SL by Gender", x = "Gender", y = "Sleep Latency (min)") +
  scale_fill_manual(values = c("lightblue", "pink")) +
  my_theme +
  theme(legend.position = "none")

# 6. SL vs. Latent Caffeine (only for simulation diagnosis)

ggplot(sim_data, aes(x = caffeine, y = sl)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "lm", color = "seagreen") +
  labs(title = "SL vs. Latent Caffeine Intake (Simulation Only)", x = "Caffeine (cups/day)",
       y = "Sleep Latency") +
  my_theme


#=============missing value simulated=====================
set.seed(123)
n_total <- nrow(sim_data)
n_target_missing <- round(0.30 * n_total)  # target ~30% missing overall

# Extract unique subjects and time points
subjects <- unique(sim_data$id)
n_subjects <- length(subjects)
times <- sort(unique(sim_data$time))
max_time <- max(times)
n_times <- length(times)

### Split missingness between MCAR and MAR
prop_mcar <- 0.6   # 60% MCAR
prop_mar <- 0.4    # 40% MAR
n_mcar <- round(prop_mcar * n_target_missing)
n_mar <- n_target_missing - n_mcar

### Create time-dependent missingness pattern
# Define a function that increases from ~10% at start to ~50% at end
time_miss_function <- function(t, max_t) {
  # Sigmoid function that starts at ~10% and ends at ~50%
  return(0.10 + 0.40 * (1 / (1 + exp(-0.20 * (t - max_t/2)))))
}

# Calculate target missingness proportion for each time point
time_props <- sapply(times, function(t) time_miss_function(t, max_time))

# Calculate how many observations to make missing at each time point
obs_per_time <- table(sim_data$time)
target_missing_per_time <- round(time_props * obs_per_time)

# Adjust to ensure we hit exactly the target number of missing values
total_targeted <- sum(target_missing_per_time)
if (total_targeted != n_target_missing) {
  # Proportionally adjust to match the target
  target_missing_per_time <- round(target_missing_per_time * (n_target_missing / total_targeted))
  
  # Handle any off-by-one errors to hit the exact target
  diff <- n_target_missing - sum(target_missing_per_time)
  if (diff > 0) {
    # Add remaining to later time points
    for (i in seq_along(times)) {
      idx <- n_times - i + 1  # Start from the end
      if (diff > 0 && target_missing_per_time[idx] < obs_per_time[idx]) {
        target_missing_per_time[idx] <- target_missing_per_time[idx] + 1
        diff <- diff - 1
      }
    }
  } else if (diff < 0) {
    # Remove from earlier time points
    for (i in seq_along(times)) {
      if (diff < 0 && target_missing_per_time[i] > 0) {
        target_missing_per_time[i] <- target_missing_per_time[i] - 1
        diff <- diff + 1
      }
    }
  }
}

# Create vectors to hold missing indices
missing_idx <- numeric(0)

### 1. First, generate MAR missingness based on anxiety and time
# Higher anxiety -> higher chance of missing
# Later time points -> higher chance of missing
for (t_idx in seq_along(times)) {
  t <- times[t_idx]
  time_rows <- which(sim_data$time == t)
  
  # Calculate MAR probability based on anxiety
  base_anxiety_effect <- 2.5 * (sim_data$anxiety[time_rows] - 0.5)
  
  # Add time effect (stronger as time progresses)
  time_effect <- 0.05 * t  # Increases linearly with time
  
  # Combined probability
  mar_probs <- 1 / (1 + exp(-(base_anxiety_effect + time_effect)))
  
  # Normalize probabilities to sum to 1
  mar_probs <- mar_probs / sum(mar_probs)
  
  # Determine how many MAR missingness to create at this time point
  n_mar_t <- round(prop_mar * target_missing_per_time[t_idx])
  
  # Sample rows to make missing based on probabilities
  if (n_mar_t > 0 && n_mar_t < length(time_rows)) {
    mar_idx_t <- sample(time_rows, size = n_mar_t, prob = mar_probs, replace = FALSE)
    missing_idx <- c(missing_idx, mar_idx_t)
  }
}

### 2. Add MCAR missingness to complete the pattern
# For each time point, add random missingness to reach the target count
for (t_idx in seq_along(times)) {
  t <- times[t_idx]
  time_rows <- which(sim_data$time == t)
  
  # Count how many are already missing at this time point
  already_missing <- sum(time_rows %in% missing_idx)
  
  # Calculate how many MCAR to add
  n_mcar_t <- target_missing_per_time[t_idx] - already_missing
  
  # Only proceed if we need to add more
  if (n_mcar_t > 0) {
    # Get eligible rows (not already marked as missing)
    eligible_rows <- setdiff(time_rows, missing_idx)
    
    # Sample rows randomly
    if (length(eligible_rows) > 0) {
      mcar_idx_t <- sample(eligible_rows, 
                           size = min(n_mcar_t, length(eligible_rows)), 
                           replace = FALSE)
      missing_idx <- c(missing_idx, mcar_idx_t)
    }
  }
}

# Apply missingness
sl_mixed <- sim_data$sl
sl_mixed[missing_idx] <- NA

# Verify total missing proportion matches target
actual_missing_prop <- mean(is.na(sl_mixed))
cat("Target missing proportion:", round(n_target_missing/n_total, 3), 
    "\nActual missing proportion:", round(actual_missing_prop, 3), "\n")

# Analyze and print missingness pattern by time
missing_by_time <- tapply(is.na(sl_mixed), sim_data$time, mean)
cat("Missingness proportion by time point:\n")
print(round(missing_by_time, 3))

# Create data frame for plotting
missing_summary <- data.frame(
  time = as.numeric(names(missing_by_time)),
  missing_percent = missing_by_time * 100
)

# Save to dataset
sim_data$sl_mixed <- sl_mixed

#===================================================================
# 1. Create summary table of missingness by day
missing_summary <- sim_data %>%
  group_by(time) %>%
  summarize(
    total_observations = n(),
    missing_count = sum(is.na(sl_mixed)),
    missing_percent = round(100 * sum(is.na(sl_mixed)) / n(), 1)
  )

# Print the summary table
print(missing_summary)

# 2. Create a grid plot showing missing pattern
# First reshape data to have one row per subject-time
missing_pattern <- sim_data %>%
  mutate(missing_status = ifelse(is.na(sl_mixed), "Missing", "Observed")) %>%
  select(id, time, missing_status) %>%
  # Ensure time is properly ordered
  mutate(time = factor(time, levels = sort(unique(time))))

# Create the missingness grid visualization
ggplot(missing_pattern, aes(x = time, y = factor(id), fill = missing_status)) +
  geom_tile(color = "white", size = 0.1) +
  scale_fill_manual(values = c("Observed" = "grey", "Missing" = "#fbb4ae")) +
  labs(
    title = "Missing Data Pattern across Subjects and Time",
    x = "Day",
    y = "Subject ID",
    fill = "Status"
  ) +
  theme_minimal() +
  theme(
    axis.text.y = element_blank(),  # Hide subject IDs for cleaner look
    panel.grid = element_blank(),
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5)
  )

# 3. Create a more compact summary showing missingness over time
missing_by_time_plot <- ggplot(missing_summary, 
                               aes(x = time, y = missing_percent)) +
  geom_col(fill = "#fbb4ae") +
  geom_text(aes(label = paste0(missing_percent, "%")), 
            vjust = -0.5, size = 3) +
  labs(
    title = "Percentage of Missing Values by Day",
    x = "Day",
    y = "Missing (%)"
  ) +
  theme_minimal() +
  ylim(0, max(missing_summary$missing_percent) * 1.1)  # Add some space for labels
#==============================mice=====================================
# Prepare the data
sim_mis <- sim_data %>%
  mutate(time_trt = time * as.numeric(treatment)) %>%
  select(sl = sl_mixed, time, treatment, time_trt, age, gender, baseline, anxiety)

# Set MICE methods
meth_pmm <- make.method(sim_mis)
meth_pmm["sl"] <- "pmm"
meth_pmm[setdiff(names(meth_pmm), "sl")] <- ""

meth_rf <- make.method(sim_mis)
meth_rf["sl"] <- "rf"
meth_rf[setdiff(names(meth_rf), "sl")] <- ""

meth_norm <- make.method(sim_mis)
meth_norm["sl"] <- "norm"
meth_norm[setdiff(names(meth_norm), "sl")] <- ""



# Run MICE
imp_pmm <- mice(sim_mis, m = 5, method = meth_pmm, seed = 123)
imp_rf  <- mice(sim_mis, m = 5, method = meth_rf, seed = 123)
imp_norm <- mice(sim_mis, m = 5, method = meth_norm, seed = 123)


# Analyze and pool
fit_pmm <- with(imp_pmm, lm(sl ~ time + treatment + time_trt + age + gender + baseline + anxiety))
fit_rf  <- with(imp_rf,  lm(sl ~ time + treatment + time_trt + age + gender + baseline + anxiety))
fit_norm <- with(imp_norm, lm(sl ~ time + treatment + time_trt + age + gender + baseline + anxiety))

pooled_pmm <- pool(fit_pmm)
pooled_rf  <- pool(fit_rf)
pooled_norm <- pool(fit_norm)



summary(pooled_pmm)
summary(pooled_rf)
summary(pooled_norm)


# ——————————————————————————————————————————————————————————————
# 5) Unified performance for all imputation methods (including coverage)

library(dplyr)


get_impu_mat <- function(imp_obj, missing_idx) {
  m <- imp_obj$m
  sapply(1:m, function(i) complete(imp_obj, i)$sl[missing_idx])
}

# Compute bias, RMSE, 90% coverage, and avg CI‑width
performance_mice <- function(imp_obj, true_vals, missing_idx, method_name) {
  imp_mat  <- get_impu_mat(imp_obj, missing_idx)
  imp_mean <- rowMeans(imp_mat)
  imp_lo   <- apply(imp_mat, 1, quantile, probs = 0.05)
  imp_hi   <- apply(imp_mat, 1, quantile, probs = 0.95)
  
  bias     <- mean(imp_mean - true_vals)
  rmse     <- sqrt(mean((imp_mean - true_vals)^2))
  coverage <- mean(true_vals >= imp_lo & true_vals <= imp_hi)
  ci_width <- mean(imp_hi - imp_lo)
  
  data.frame(
    method   = method_name,
    bias     = bias,
    rmse     = rmse,
    coverage = coverage,
    ci_width = ci_width
  )
}

# True hold‑out values
true_vals <- sim_data$sl[missing_idx]

# Compute performance for each MICE variant
perf_pmm  <- performance_mice(imp_pmm,  true_vals, missing_idx, "MICE‑PMM")
perf_rf   <- performance_mice(imp_rf,   true_vals, missing_idx, "MICE‑RF")
perf_norm <- performance_mice(imp_norm, true_vals, missing_idx, "MICE‑Norm")


#===================gp====================================
# ——— prerequisites: sim_data exists, with sl_mixed already set ———

# 0) Define obs/mis and design matrices
obs_idx <- which(!is.na(sim_data$sl_mixed))
mis_idx <- which( is.na(sim_data$sl_mixed))

Z_full <- model.matrix(~ time + treatment, data = sim_data)
X_full <- model.matrix(
  ~ time * treatment + age + gender + baseline + anxiety,
  data = sim_data
)

X_obs   <- X_full[obs_idx, ]
X_mis   <- X_full[mis_idx, ]
Z_obs   <- Z_full[obs_idx, ]
Z_mis   <- Z_full[mis_idx, ]
y_obs   <- sim_data$sl_mixed[obs_idx]
trt_obs <- as.integer(sim_data$treatment)[obs_idx] - 1
trt_mis <- as.integer(sim_data$treatment)[mis_idx] - 1

# 1) Pick M inducing points (e.g. sqrt rule)
M <- floor(sqrt(length(obs_idx)))  # or choose 10–20% of N_obs
set.seed(2025)
inducing_idx  <- sample(obs_idx, M)
Z_inducing    <- X_full[inducing_idx, ]
trt_inducing  <- as.integer(sim_data$treatment)[inducing_idx] - 1

# 2) Pack stan_data
stan_data <- list(
  N_obs        = length(obs_idx),
  N_mis        = length(mis_idx),
  D            = ncol(X_full),
  P            = ncol(Z_full),
  M            = M,
  X_obs        = X_obs,
  X_mis        = X_mis,
  Z_obs        = Z_obs,
  Z_mis        = Z_mis,
  y_obs        = y_obs,
  trt_obs      = trt_obs,
  trt_mis      = trt_mis,
  Z_inducing   = Z_inducing,
  trt_inducing = trt_inducing,
  alpha_prior  = 5,    # increased from 5
  rho_prior    = 10,    # increased from 10
  sigma_prior  = 5,     # keep the same
  jitter_val   = 1e-5   # increased from 1e-6
)

# 3) Compile & fit
library(rstan); library(bayesplot)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

gp_mod <- stan_model(file = "C:/Users/Chang/OneDrive/PHP 2530/Final/sparse_gp.stan")
fit_gp <- sampling(gp_mod, data = stan_data,
                   iter = 2000, warmup = 1000,
                   chains = 4, cores = 4, refresh = 50,
                   control = list(adapt_delta = 0.95))

# 4) Convergence checks
print(fit_gp, pars = c("beta","alpha","rho","sigma"), probs=c(.1,.5,.9))

# base‐R traceplots for the first four betas + hyper‑parameters
traceplot(fit_gp,
          pars      = c("beta[1]","beta[2]","beta[3]","beta[4]",
                        "alpha","rho","sigma"),
          inc_warmup = FALSE)

# bayesplot version
post_array <- as.array(fit_gp)
mcmc_trace(post_array,
           pars = c("beta[1]","beta[2]","beta[3]","beta[4]",
                    "alpha","rho","sigma"),
           facet_args = list(ncol = 4, strip.position = "top"))

# 5) Imputation results
draws         <- extract(fit_gp, "y_mis_pred")[[1]]
mean_imp      <- colMeans(draws)
lower_imp     <- apply(draws, 2, quantile, probs=0.05)
upper_imp     <- apply(draws, 2, quantile, probs=0.95)

sim_data$sl_imputed <- sim_data$sl_mixed
sim_data$sl_imputed[mis_idx] <- mean_imp

# 6) Performance metrics
true_vals <- sim_data$sl[mis_idx]
bias      <- mean(mean_imp - true_vals)
rmse      <- sqrt(mean((mean_imp - true_vals)^2))
coverage  <- mean(true_vals >= lower_imp & true_vals <= upper_imp)
ci_width  <- mean(upper_imp - lower_imp)

performance_gp <- data.frame(
  method   = "Sparse GP",
  bias     = bias,
  rmse     = rmse,
  coverage = coverage,
  ci_width = ci_width
)
print(performance_gp)

# Combine with your Sparse GP result
performance_all <- bind_rows(
  performance_gp,
  perf_pmm,
  perf_rf,
  perf_norm
)

print(performance_all)



