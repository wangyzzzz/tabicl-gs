args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag) {
  idx <- match(flag, args)
  if (is.na(idx) || idx == length(args)) {
    stop(paste("Missing argument:", flag))
  }
  args[idx + 1]
}

suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(sommer))
suppressPackageStartupMessages(library(BGLR))

model_name <- get_arg("--model")
train_x_path <- get_arg("--train-x")
train_y_path <- get_arg("--train-y")
test_x_path <- get_arg("--test-x")
pred_out <- get_arg("--pred-out")
meta_out <- get_arg("--meta-out")
seed <- as.integer(get_arg("--seed"))
sommer_method <- if ("--sommer-method" %in% args) get_arg("--sommer-method") else "mmer"
beta_out <- if ("--beta-out" %in% args) get_arg("--beta-out") else NA
bandwidth_scale <- if ("--bandwidth-scale" %in% args) as.numeric(get_arg("--bandwidth-scale")) else 1.0

set.seed(seed)

train_x <- as.matrix(read.csv(train_x_path, header = FALSE, check.names = FALSE))
train_y <- as.numeric(read.csv(train_y_path, header = FALSE, check.names = FALSE)[, 1])
test_x <- as.matrix(read.csv(test_x_path, header = FALSE, check.names = FALSE))

fill_na <- function(train_mat, test_mat) {
  means <- colMeans(train_mat, na.rm = TRUE)
  means[is.na(means)] <- 0
  train_idx <- which(is.na(train_mat), arr.ind = TRUE)
  test_idx <- which(is.na(test_mat), arr.ind = TRUE)
  if (nrow(train_idx) > 0) {
    train_mat[train_idx] <- means[train_idx[, 2]]
  }
  if (nrow(test_idx) > 0) {
    test_mat[test_idx] <- means[test_idx[, 2]]
  }
  list(train = train_mat, test = test_mat)
}

filled <- fill_na(train_x, test_x)
train_x <- filled$train
test_x <- filled$test

predict_from_sommer_mmer <- function(train_x, train_y, test_x) {
  all_x <- rbind(train_x, test_x)
  allele_freq <- colMeans(train_x, na.rm = TRUE) / 2
  centered <- sweep(all_x, 2, 2 * allele_freq, "-")
  denom <- 2 * sum(allele_freq * (1 - allele_freq))
  if (!is.finite(denom) || denom <= 0) {
    denom <- ncol(train_x)
  }
  G <- tcrossprod(centered) / denom
  ids <- as.character(seq_len(nrow(all_x)))
  rownames(G) <- ids
  colnames(G) <- ids
  df <- data.frame(id = factor(ids, levels = ids), y = c(train_y, rep(NA_real_, nrow(test_x))))
  fit <- sommer::mmer(y ~ 1, random = ~ sommer::vsr(id, Gu = G), rcov = ~ units, data = df, verbose = FALSE)
  intercept <- fit$Beta$Estimate[fit$Beta$Effect == "(Intercept)"][1]
  if (!is.finite(intercept)) {
    intercept <- 0
  }
  random_block <- fit$U[[1]]
  random_effect <- random_block[[1]]
  all_pred <- intercept + as.numeric(random_effect[ids])
  all_pred[(nrow(train_x) + 1):nrow(all_x)]
}

predict_from_sommer_mmes <- function(train_x, train_y, test_x) {
  all_x <- rbind(train_x, test_x)
  allele_freq <- colMeans(train_x, na.rm = TRUE) / 2
  centered <- sweep(all_x, 2, 2 * allele_freq, "-")
  denom <- 2 * sum(allele_freq * (1 - allele_freq))
  if (!is.finite(denom) || denom <= 0) {
    denom <- ncol(train_x)
  }
  G <- tcrossprod(centered) / denom
  ids <- as.character(seq_len(nrow(all_x)))
  rownames(G) <- ids
  colnames(G) <- ids
  df <- data.frame(id = factor(ids, levels = ids), y = c(train_y, rep(NA_real_, nrow(test_x))))
  fit <- sommer::mmes(
    y ~ 1,
    random = ~ sommer::vsm(sommer::ism(id), Gu = G),
    rcov = ~ units,
    data = df,
    verbose = FALSE,
    naMethodY = "include"
  )
  intercept <- as.numeric(fit$b[1, 1])
  random_effect <- fit$u[ids, 1]
  all_pred <- intercept + as.numeric(random_effect)
  all_pred[(nrow(train_x) + 1):nrow(all_x)]
}

predict_from_bglr <- function(train_x, train_y, test_x, model_name) {
  eta_model <- switch(
    model_name,
    "BayesA" = "BayesA",
    "BayesB" = "BayesB",
    "BayesLasso" = "BL",
    stop(paste("Unsupported BGLR model:", model_name))
  )
  save_prefix <- tempfile(pattern = "bglr_")
  fit <- BGLR::BGLR(
    y = train_y,
    ETA = list(list(X = train_x, model = eta_model)),
    nIter = 10000,
    burnIn = 1000,
    thin = 5,
    verbose = FALSE,
    saveAt = save_prefix
  )
  list(
    predictions = as.numeric(fit$mu + test_x %*% fit$ETA[[1]]$b),
    beta = as.numeric(fit$ETA[[1]]$b)
  )
}

compute_rbf_kernel <- function(train_x, test_x, bandwidth_scale = 1.0) {
  all_x <- rbind(train_x, test_x)
  train_means <- colMeans(train_x, na.rm = TRUE)
  train_sds <- apply(train_x, 2, sd, na.rm = TRUE)
  train_sds[!is.finite(train_sds) | train_sds <= 1e-8] <- 1.0
  centered <- sweep(all_x, 2, train_means, "-")
  scaled <- sweep(centered, 2, train_sds, "/")
  dist_sq <- as.matrix(stats::dist(scaled, method = "euclidean"))^2
  n_train <- nrow(train_x)
  if (n_train > 1) {
    train_dist_sq <- dist_sq[seq_len(n_train), seq_len(n_train), drop = FALSE]
    upper_vals <- train_dist_sq[upper.tri(train_dist_sq)]
    median_sq <- stats::median(upper_vals[is.finite(upper_vals) & upper_vals > 0], na.rm = TRUE)
  } else {
    median_sq <- NA_real_
  }
  if (!is.finite(median_sq) || median_sq <= 0) {
    median_sq <- max(ncol(train_x), 1)
  }
  gamma <- 1.0 / (bandwidth_scale * median_sq)
  exp(-gamma * dist_sq)
}

predict_from_bglr_rkhs <- function(train_x, train_y, test_x, bandwidth_scale = 1.0) {
  K_all <- compute_rbf_kernel(train_x, test_x, bandwidth_scale = bandwidth_scale)
  y_all <- c(train_y, rep(NA_real_, nrow(test_x)))
  save_prefix <- tempfile(pattern = "bglr_rkhs_")
  fit <- BGLR::BGLR(
    y = y_all,
    ETA = list(list(K = K_all, model = "RKHS")),
    nIter = 10000,
    burnIn = 1000,
    thin = 5,
    verbose = FALSE,
    saveAt = save_prefix
  )
  list(
    predictions = as.numeric(fit$yHat[(nrow(train_x) + 1):nrow(K_all)]),
    beta = NULL
  )
}

result <- switch(
  model_name,
  "GBLUP" = switch(
    sommer_method,
    "mmer" = list(predictions = predict_from_sommer_mmer(train_x, train_y, test_x), beta = NULL),
    "mmes" = list(predictions = predict_from_sommer_mmes(train_x, train_y, test_x), beta = NULL),
    stop(paste("Unsupported sommer method:", sommer_method))
  ),
  "BayesA" = predict_from_bglr(train_x, train_y, test_x, model_name),
  "BayesB" = predict_from_bglr(train_x, train_y, test_x, model_name),
  "BayesLasso" = predict_from_bglr(train_x, train_y, test_x, model_name),
  "RKHS" = predict_from_bglr_rkhs(train_x, train_y, test_x, bandwidth_scale = bandwidth_scale),
  stop(paste("Unsupported model:", model_name))
)

predictions <- result$predictions

write.table(predictions, file = pred_out, sep = ",", row.names = FALSE, col.names = FALSE)
if (!is.na(beta_out) && !is.null(result$beta)) {
  write.table(result$beta, file = beta_out, sep = ",", row.names = FALSE, col.names = FALSE)
}
write_json(
  list(
    model = model_name,
    sommer_method = sommer_method,
    bandwidth_scale = bandwidth_scale,
    n_train = nrow(train_x),
    n_test = nrow(test_x),
    has_beta = !is.null(result$beta),
    device = "R"
  ),
  path = meta_out,
  auto_unbox = TRUE,
  pretty = TRUE
)
