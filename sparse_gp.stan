functions {
  // Squared‐exponential (RBF) kernel between two row‐vectors with ARD length‐scales
  real cov_rbf_one(row_vector x1, row_vector x2,
                   real alpha, vector rho) {
    // Fixed the element-wise division by ensuring type compatibility
    row_vector[cols(x1)] diff = x1 - x2;
    row_vector[cols(x1)] scaled_diff;
    for (d in 1:cols(x1))
      scaled_diff[d] = diff[d] / rho[d];
    
    return square(alpha) * exp(-0.5 * sum(square(scaled_diff)));
  }

  // Matrix form: continuous‐covariate kernel
  matrix cov_rbf(matrix X1, matrix X2,
                 real alpha, vector rho) {
    int n1 = rows(X1), n2 = rows(X2);
    matrix[n1,n2] K;
    for (i in 1:n1) {
      for (j in 1:n2) {
        K[i,j] = cov_rbf_one(X1[i], X2[j], alpha, rho);
      }
    }
    return K;
  }

  // Single‐pair treatment kernel
  real cov_trt_one(int a, int b, real alpha_trt) {
    return (a == b) ? square(alpha_trt) : 0;
  }

  // Matrix form: treatment‐specific kernel
  matrix cov_treatment(int[] trt1, int[] trt2, real alpha_trt) {
    int n1 = size(trt1), n2 = size(trt2);
    matrix[n1,n2] K;
    for (i in 1:n1) {
      for (j in 1:n2) {  // Fixed semicolon to colon
        K[i,j] = cov_trt_one(trt1[i], trt2[j], alpha_trt);
      }
    }
    return K;
  }
}

data {
  int<lower=1>   N_obs;            // # observed points
  int<lower=1>   N_mis;            // # to predict/impute
  int<lower=1>   D;                // dim of kernel inputs
  int<lower=1>   P;                // dim of linear mean
  int<lower=1>   M;                // # inducing points
  int<lower=0,upper=2> trt_obs[N_obs];
  int<lower=0,upper=2> trt_mis[N_mis];
  matrix[N_obs, D] X_obs;
  matrix[N_mis, D] X_mis;
  matrix[M, D]     Z_inducing;
  int<lower=0,upper=2> trt_inducing[M];
  matrix[N_obs, P] Z_obs;
  matrix[N_mis, P] Z_mis;
  vector[N_obs]    y_obs;
  real<lower=0>    alpha_prior;
  real<lower=0>    rho_prior;
  real<lower=0>    sigma_prior;
  real<lower=0>    jitter_val;
}

transformed data {
  // jitter for numerical stability
  matrix[M,M] jitter = diag_matrix(rep_vector(jitter_val, M));
}

parameters {
  real<lower=0>        alpha;        // GP amplitude
  vector<lower=0>[D]   rho;          // ARD length‐scales
  real<lower=0>        alpha_trt;    // treatment‐kernel amp
  real<lower=0>        sigma;        // obs noise SD
  vector[P]            beta;         // linear coefs
  vector[M]            u;            // inducing‐point latents
}

transformed parameters {
  // Build and factor K_mm
  matrix[M,M] K_time = cov_rbf(Z_inducing, Z_inducing, alpha, rho);
  matrix[M,M] K_trt  = cov_treatment(trt_inducing, trt_inducing, alpha_trt);
  matrix[M,M] Kmm    = K_time + K_trt + jitter;
  matrix[M,M] L_Kmm  = cholesky_decompose(Kmm);

  // Cross‐covariances for observed
  matrix[N_obs, M] K_time_obs = cov_rbf(X_obs, Z_inducing, alpha, rho);
  matrix[N_obs, M] K_trt_obs  = cov_treatment(trt_obs, trt_inducing, alpha_trt);
  matrix[N_obs, M] Knm_obs     = K_time_obs + K_trt_obs;

  // Precompute Kmm⁻¹ * u
  vector[M] Kmm_inv_u = mdivide_left_spd(Kmm, u);
}

model {
  // Priors
  beta       ~ normal(0, 5);
  alpha      ~ cauchy(0, alpha_prior);
  alpha_trt  ~ cauchy(0, alpha_prior);
  rho        ~ gamma(2, 1.0 / rho_prior);
  sigma      ~ cauchy(0, sigma_prior);

  // GP prior on u
  u ~ multi_normal_cholesky(rep_vector(0, M), L_Kmm);

  // Marginal likelihood
  {
    vector[N_obs] mu_obs = Z_obs * beta + Knm_obs * Kmm_inv_u;
    y_obs ~ normal(mu_obs, sigma);
  }
}

generated quantities {
  // Cross‐covariances for missing
  matrix[N_mis, M] K_time_mis = cov_rbf(X_mis, Z_inducing, alpha, rho);
  matrix[N_mis, M] K_trt_mis  = cov_treatment(trt_mis, trt_inducing, alpha_trt);
  matrix[N_mis, M] Knm_mis     = K_time_mis + K_trt_mis;

  // Predictive mean
  vector[N_mis] mu_mis = Z_mis * beta
                      + Knm_mis * mdivide_left_spd(Kmm, u);

  // Predictive variance
  matrix[M, N_mis] Kmm_inv_Knm_mis = mdivide_left_spd(Kmm, Knm_mis');
  vector[N_mis] var_mis;
  for (i in 1:N_mis) {
    row_vector[D] x_i = X_mis[i];
    real kxx = cov_rbf_one(x_i, x_i, alpha, rho)
             + cov_trt_one(trt_mis[i], trt_mis[i], alpha_trt);
    var_mis[i] = square(sigma) + kxx
               - dot_product(Knm_mis[i], Kmm_inv_Knm_mis[,i]);
  }

  // Posterior predictive draws - Fixed type issue
  vector[N_mis] y_mis_pred;
  for (i in 1:N_mis)
    y_mis_pred[i] = normal_rng(mu_mis[i], sqrt(var_mis[i]));

  // Log‐likelihood for loo/WAIC
  vector[N_obs] log_lik;
  {
    vector[N_obs] mu = Z_obs * beta + Knm_obs * mdivide_left_spd(Kmm, u);
    for (i in 1:N_obs)
      log_lik[i] = normal_lpdf(y_obs[i] | mu[i], sigma);
  }
}


