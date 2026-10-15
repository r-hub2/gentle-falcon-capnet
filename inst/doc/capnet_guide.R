## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)
glmnet_ok <- requireNamespace("glmnet", quietly = TRUE)
xtable_ok <- requireNamespace("xtable", quietly = TRUE)
if (!glmnet_ok || !xtable_ok) knitr::opts_chunk$set(eval = FALSE)

library(lbfgs)
library(ggplot2)
library(xts)
library(capnet)
if (glmnet_ok) library(glmnet)
if (xtable_ok) library(xtable)
options(scipen = 0)
convergence.table <- NULL

## ----data---------------------------------------------------------------------
set.seed(123)
example <- simulate_capnet_data()
X <- example$X
y <- example$y
n <- nrow(X)
p <- ncol(X)
L <- example$L
guide_lambda <- exp(seq(1, -5, length.out = 25))
guide_alpha <- seq(0, 1, length.out = 5)
maxx <- t(as.matrix(apply(X, 2, function(x) {
  mx <- max(x)
  mn <- min(x)
  if (abs(mx) > abs(mn)) mx else mn
})))
beta_init <- rep(0, p + 1)

## ----no-penalty---------------------------------------------------------------
set.seed(1)
capnet.cv1 <- cv_capnet(
  X, y, 0, L, lambda = guide_lambda, alpha = guide_alpha,
  standardize = FALSE
)
alpha1 <- capnet.cv1$best_alpha
lambda1 <- capnet.cv1$best_lambda
capnet.mod1 <- capnet(X, y,
  alpha = alpha1, lambda = lambda1,
  gamma = 0, L = L, standardize = FALSE
)

## ----results='asis', echo=FALSE-----------------------------------------------
cat(sprintf("Best alpha value: %.3f | Best lambda value %.4f", alpha1, lambda1))

## ----penalty------------------------------------------------------------------
set.seed(2)
capnet.cv2 <- cv_capnet(
  X, y, 1, L, lambda = guide_lambda, alpha = guide_alpha,
  standardize = FALSE
)
alpha2 <- capnet.cv2$best_alpha
lambda2 <- capnet.cv2$best_lambda
capnet.mod2 <- capnet(X, y,
  alpha = alpha2, lambda = lambda2,
  gamma = 0.5, L = L, standardize = FALSE
)

## ----results='asis', echo=FALSE-----------------------------------------------
cat(sprintf("Best alpha value: %.3f | Best lambda value %.4f", alpha2, lambda2))

## ----weights, fig.width=9, fig.height=5, out.width='100%'---------------------
plots <- plot_redistribution(capnet.mod1, capnet.mod2, plot = "raw")
plots$beta
plots$contribution

## ----runtime, echo=FALSE------------------------------------------------------
print("System runtime for unconstrained 5-fold cross-validation")
print(system.time(cv_capnet(
  X, y, gamma = 0, L = L, lambda = guide_lambda,
  alpha = guide_alpha, standardize = FALSE
)))
print("System runtime for constrained 5-fold cross-validation")
print(system.time(cv_capnet(
  X, y, gamma = 1, L = L, lambda = guide_lambda,
  alpha = guide_alpha, standardize = FALSE
)))
print("System runtime for constrained model fit")
print(system.time(capnet(X, y,
        alpha = 0.5, lambda = 0.1, gamma = 10, L = L,
        standardize = FALSE
      )))

## ----experiment, results="asis"-----------------------------------------------
for (i in seq(-5, 10)) {
  # Regular model
  model1 <- capnet(X, y,
    alpha = 0.5, lambda = 0.01,
    gamma = 10**i, L = L, z = X, standardize = TRUE, multiplier = 1
  )
  convergence1 <- model1$convergence

  # Larger lambda model
  model2 <- capnet(X, y,
    alpha = 0.5, lambda = 0.5,
    gamma = 10**i, L = L, z = X, standardize = TRUE, multiplier = 1
  )
  convergence2 <- model2$convergence

  # No elastic net penalty model
  model3 <- capnet(X, y,
    alpha = 0.5, lambda = 0,
    gamma = 10**i, L = L, z = X, standardize = TRUE, multiplier = 1
  )
  convergence3 <- model3$convergence

  # Ridge regression model
  model4 <- capnet(X, y,
    alpha = 0, lambda = 0.01,
    gamma = 10**i, L = L, z = X, standardize = TRUE, multiplier = 1
  )
  convergence4 <- model4$convergence

  # LASSO regression model
  model5 <- capnet(X, y,
    alpha = 1, lambda = 0.01,
    gamma = 10**i, L = L, z = X, standardize = TRUE, multiplier = 1
  )
  convergence5 <- model5$convergence

  # Large multiplier model
  model6 <- capnet(X, y,
    alpha = 0.5, lambda = 0.01,
    gamma = 10**i, L = L, z = X, standardize = TRUE, multiplier = 100
  )
  convergence6 <- model6$convergence

  # Small L model
  model7 <- capnet(X, y,
    alpha = 0.5, lambda = 0.01,
    gamma = 10**i, L = L / 2, z = X, standardize = TRUE, multiplier = 1
  )
  convergence7 <- model7$convergence

  # Single max-row model
  model8 <- capnet(X, y,
    alpha = 0.5, lambda = 0.01,
    gamma = 10**i, L = L, z = maxx, standardize = TRUE, multiplier = 1
  )
  convergence8 <- model8$convergence

  convergence.table <- rbind(
    convergence.table,
    c(
      convergence1, convergence2,
      convergence3, convergence4,
      convergence5, convergence6,
      convergence7, convergence8
    )
  )
}

## ----table, results='asis', echo=FALSE, message=FALSE, warning=FALSE----------
convergence.table <- as.data.frame(convergence.table)
convergence.table[] <- lapply(convergence.table, as.character)
convergence.table <- cbind(10**seq(-5, 10), convergence.table)
colnames(convergence.table) <- c(
  "Gamma",
  "Base",
  "Large Lambda",
  "No Elastic Net",
  "Ridge",
  "LASSO",
  "Large Multiplier",
  "Small L",
  "Single Max Row"
)

if (knitr::is_latex_output()) {
  print(
    xtable::xtable(convergence.table, display = c(rep("g", 10))),
    math.style.exponents = TRUE,
    sanitize.colnames.function = identity,
    comment = FALSE
  )
} else {
  knitr::kable(convergence.table, format = "html")
}

## ----lambda-values------------------------------------------------------------
K <- 5L
n_runs <- 10L
lambda_grid <- 10^seq(-11, -1, length.out = 6)

run_one <- function(i) {
  set.seed(i)
  cv0 <- cv_capnet(X, y,
    gamma = 0, L = L,
    alpha = 0.5, lambda = lambda_grid,
    z = maxx, K = K, standardize = TRUE, multiplier = 1
  )
  cv01 <- cv_capnet(X, y,
    gamma = 0.1, L = L,
    alpha = 0.5, lambda = lambda_grid,
    z = maxx, K = K, standardize = TRUE, multiplier = 1
  )
  cv100 <- cv_capnet(X, y,
    gamma = 100, L = L,
    alpha = 0.5, lambda = lambda_grid,
    z = maxx, K = K, standardize = TRUE, multiplier = 1
  )

  fit0 <- capnet(X, y,
    alpha = 0.5, lambda = cv0$best_lambda,
    gamma = 0, L = L, z = maxx, standardize = TRUE, multiplier = 1
  )
  fit01 <- capnet(X, y,
    alpha = 0.5, lambda = cv01$best_lambda,
    gamma = 0.1, L = L, z = maxx, standardize = TRUE, multiplier = 1
  )
  fit100 <- capnet(X, y,
    alpha = 0.5, lambda = cv100$best_lambda,
    gamma = 100, L = L, z = maxx, standardize = TRUE, multiplier = 1
  )

  list(
    lambda0 = cv0$best_lambda, beta0 = as.numeric(fit0$beta),
    lambda01 = cv01$best_lambda, beta01 = as.numeric(fit01$beta),
    lambda100 = cv100$best_lambda, beta100 = as.numeric(fit100$beta)
  )
}

results <- lapply(seq_len(n_runs), run_one)

lambdas0 <- sapply(results, `[[`, "lambda0")
lambdas01 <- sapply(results, `[[`, "lambda01")
lambdas100 <- sapply(results, `[[`, "lambda100")

betas0 <- do.call(rbind, lapply(results, `[[`, "beta0"))
betas01 <- do.call(rbind, lapply(results, `[[`, "beta01"))
betas100 <- do.call(rbind, lapply(results, `[[`, "beta100"))

## ----plot-lambda, echo=FALSE, fig.width=9, fig.height=5, out.width='100%'-----
best_lambdas <- as.data.frame(rbind(
  cbind(lambdas0, 0),
  cbind(lambdas01, 0.1),
  cbind(lambdas100, 100)
))
names(best_lambdas) <- c("lambda", "gamma")

ggplot(best_lambdas) +
  geom_density(aes(x = lambda, color = as.factor(gamma))) +
  scale_x_continuous(trans = "log10") +
  labs(x = "Lambda", y = "Density", color = "gamma") +
  theme_minimal()

## ----beta-bands, echo=FALSE, fig.width=9, fig.height=5, out.width='100%'------
.beta_summary <- function(betas, gamma_label) {
  data.frame(
    feature = seq_len(ncol(betas)),
    mean    = colMeans(betas),
    lo      = apply(betas, 2, min),
    hi      = apply(betas, 2, max),
    gamma   = gamma_label
  )
}

beta_df <- rbind(
  .beta_summary(betas0, "gamma = 0"),
  .beta_summary(betas01, "gamma = 0.1"),
  .beta_summary(betas100, "gamma = 100")
)
beta_df$gamma <- factor(beta_df$gamma,
  levels = c("gamma = 0", "gamma = 0.1", "gamma = 100")
)

ggplot(beta_df, aes(x = feature, y = mean, color = gamma, fill = gamma)) +
  geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.15, color = NA) +
  geom_line(linewidth = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  scale_x_continuous(breaks = seq_len(max(beta_df$feature))) +
  labs(x = "Feature", y = "Coefficient", color = NULL, fill = NULL) +
  theme_bw(base_size = 14) +
  theme(legend.position = "top", panel.grid.minor = element_blank())

## ----helpers, include=FALSE---------------------------------------------------
simulate_simple_proxy <- function(
  n = 1200, p = 8,
  sparsity = 0.5,
  z_sd = 1,
  eps_sd = 1,
  unstable_idx = NULL,
  unstable_frac = 0.25,
  a_sd = 0.3,
  sigma_low = 1.0,
  sigma_high = 8.0,
  high_prob = 0.10,
  seed = 2025
) {
  set.seed(seed)
  Z <- matrix(rnorm(n * p, sd = z_sd), n, p)
  beta <- rnorm(p)
  if (sparsity > 0) beta[sample(1:p, floor(sparsity * p))] <- 0
  y <- drop(Z %*% beta + rnorm(n, sd = eps_sd))
  a <- exp(rnorm(p, mean = 0, sd = a_sd))
  mean_part <- sweep(Z, 2, a, `*`)
  if (is.null(unstable_idx)) {
    k <- max(1, round(unstable_frac * p))
    unstable_idx <- sort(sample(seq_len(p), k, replace = FALSE))
  }
  stable_idx <- setdiff(seq_len(p), unstable_idx)
  Eta <- matrix(rnorm(n * p, sd = sigma_low), n, p)
  if (length(unstable_idx) > 0) {
    flips <- matrix(
      rbinom(n * length(unstable_idx), 1, high_prob), n,
      length(unstable_idx)
    )
    sd_mat <- sigma_low + flips * (sigma_high - sigma_low)
    Eta[, unstable_idx] <- matrix(
      rnorm(n * length(unstable_idx)), n,
      length(unstable_idx)
    ) * sd_mat
  }
  X <- mean_part + Eta
  list(
    X = X, y = y, Z = Z, beta = beta, a = a,
    unstable_idx = unstable_idx, stable_idx = stable_idx
  )
}

prepare_data <- function(data_source = c("quickstart", "proxy")) {
  data_source <- match.arg(data_source)
  if (data_source == "quickstart") {
    data(QuickStartExample, package = "glmnet")
    X <- QuickStartExample$x[, 1:5]
    y <- QuickStartExample$y
  } else {
    sim <- simulate_simple_proxy(
      n = 1200, p = 8, unstable_frac = 0.10,
      sigma_low = 1.0, sigma_high = 8.0, high_prob = 0.10, seed = 2025
    )
    X <- sim$X
    y <- sim$y
  }
  y <- scale(y)
  set.seed(123)
  shuffle <- sample(seq_len(nrow(X)))
  X <- X[shuffle, ]
  y <- y[shuffle, , drop = FALSE]
  n <- nrow(X)
  split <- floor(n * 0.8)
  train_X <- X[1:split, ]
  train_y <- y[1:split, , drop = FALSE]
  test_X <- X[(split + 1):n, ]
  test_y <- y[(split + 1):n, , drop = FALSE]
  gamma <- 100
  if (data_source == "quickstart") {
    L <- runif(ncol(X), 0.8, 1.2)
  } else {
    kappa <- 0.1
    L <- kappa * apply(abs(train_X), 2, function(v) stats::quantile(v, 0.99))
  }
  list(
    train_X = train_X, train_y = train_y, test_X = test_X,
    test_y = test_y, L = L, gamma = gamma
  )
}

fit_models <- function(train_X, train_y, test_X, L, gamma) {
  alphas <- c(0, 0.2, 0.4, 0.6, 0.8, 1)
  best_alpha <- 0
  best_lambda <- 0
  best_loss <- Inf
  for (alpha in alphas) {
    enet.cv <- cv.glmnet(train_X, train_y, alpha = alpha)
    lambda_enet <- enet.cv$lambda.min
    enet.x <- glmnet(train_X, train_y, alpha = alpha, lambda = lambda_enet)
    loss <- sum(predict(enet.x, train_X) - train_y)^2
    if (loss < best_loss) {
      best_alpha <- alpha
      best_lambda <- lambda_enet
      best_loss <- loss
    }
  }
  best_enet <- glmnet(train_X, train_y,
    alpha = best_alpha,
    lambda = best_lambda
  )
  enet_beta0 <- coef(best_enet)[1]
  enet_beta <- coef(best_enet)[-1]
  original_pred_mean <- sum(enet_beta * colMeans(train_X))
  for (i in seq_along(enet_beta)) {
    if (abs(enet_beta[i] * max(train_X[, i])) > L[i]) {
      enet_beta[i] <- (L[i] / max(train_X[, i])) * sign(enet_beta[i])
    }
  }
  new_pred_mean <- sum(enet_beta * colMeans(train_X))
  enet_beta0 <- enet_beta0 + (original_pred_mean - new_pred_mean)
  cap1 <- capnet(train_X, train_y, L,
    lambda = best_lambda, alpha = best_alpha,
    gamma = gamma, standardize = FALSE
  )
  cap2 <- walk_capnet(train_X, train_y, L, test_X,
    lambda = best_lambda, alpha = best_alpha,
    gamma = gamma, standardize = FALSE
  )
  list(
    enet = list(beta0 = enet_beta0, beta = enet_beta),
    cap1 = list(beta0 = cap1$a0, beta = cap1$beta),
    cap2 = cap2
  )
}

mse_from_beta <- function(X, y, beta0, beta) {
  pred <- as.numeric(beta0 + X %*% beta)
  mean((pred - as.numeric(y))^2)
}

mse_from_pred <- function(pred, y) {
  mean((as.numeric(pred) - as.numeric(y))^2)
}

## ----quickstart---------------------------------------------------------------
ds_qs <- prepare_data("quickstart")
set.seed(123)
fits_qs <- fit_models(
  ds_qs$train_X, ds_qs$train_y, ds_qs$test_X, ds_qs$L,
  ds_qs$gamma
)

qs_enet_is <- mse_from_beta(
  ds_qs$train_X, ds_qs$train_y, fits_qs$enet$beta0,
  fits_qs$enet$beta
)
qs_cap1_is <- mse_from_beta(
  ds_qs$train_X, ds_qs$train_y, fits_qs$cap1$beta0,
  fits_qs$cap1$beta
)

qs_enet_oos <- mse_from_beta(
  ds_qs$test_X, ds_qs$test_y, fits_qs$enet$beta0,
  fits_qs$enet$beta
)
qs_cap1_oos <- mse_from_beta(
  ds_qs$test_X, ds_qs$test_y, fits_qs$cap1$beta0,
  fits_qs$cap1$beta
)
qs_cap2_oos <- mse_from_pred(fits_qs$cap2$predictions, ds_qs$test_y)

report_qs <- data.frame(
  Model = c("Elastic Net (post-fit cap)", "Optimized Cap", "Walk Cap"),
  IS = c(qs_enet_is, qs_cap1_is, NA),
  OOS = c(qs_enet_oos, qs_cap1_oos, qs_cap2_oos)
)
knitr::kable(report_qs, digits = 6)

## ----proxy--------------------------------------------------------------------
ds_px <- prepare_data("proxy")
set.seed(123)
fits_px <- fit_models(
  ds_px$train_X, ds_px$train_y, ds_px$test_X, ds_px$L,
  ds_px$gamma
)

px_enet_is <- mse_from_beta(
  ds_px$train_X, ds_px$train_y, fits_px$enet$beta0,
  fits_px$enet$beta
)
px_cap1_is <- mse_from_beta(
  ds_px$train_X, ds_px$train_y, fits_px$cap1$beta0,
  fits_px$cap1$beta
)

px_enet_oos <- mse_from_beta(
  ds_px$test_X, ds_px$test_y, fits_px$enet$beta0,
  fits_px$enet$beta
)
px_cap1_oos <- mse_from_beta(
  ds_px$test_X, ds_px$test_y, fits_px$cap1$beta0,
  fits_px$cap1$beta
)
px_cap2_oos <- mse_from_pred(fits_px$cap2$predictions, ds_px$test_y)

report_px <- data.frame(
  Model = c("Elastic Net (post-fit cap)", "Optimized Cap", "Walk Cap"),
  IS = c(px_enet_is, px_cap1_is, NA),
  OOS = c(px_enet_oos, px_cap1_oos, px_cap2_oos)
)
knitr::kable(report_px, digits = 6)

