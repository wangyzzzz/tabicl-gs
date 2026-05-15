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

x_path <- get_arg("--x")
y_path <- get_arg("--y")
out_path <- get_arg("--out")
seed <- as.integer(get_arg("--seed"))
sommer_method <- if ("--sommer-method" %in% args) get_arg("--sommer-method") else "mmer"

set.seed(seed)

X <- as.matrix(read.csv(x_path, header = FALSE, check.names = FALSE))
y <- as.numeric(read.csv(y_path, header = FALSE, check.names = FALSE)[, 1])

fill_na_single <- function(mat) {
  means <- colMeans(mat, na.rm = TRUE)
  means[is.na(means)] <- 0
  idx <- which(is.na(mat), arr.ind = TRUE)
  if (nrow(idx) > 0) {
    mat[idx] <- means[idx[, 2]]
  }
  mat
}

build_g_matrix <- function(X) {
  X <- fill_na_single(X)
  allele_freq <- colMeans(X, na.rm = TRUE) / 2
  centered <- sweep(X, 2, 2 * allele_freq, "-")
  denom <- 2 * sum(allele_freq * (1 - allele_freq))
  if (!is.finite(denom) || denom <= 0) {
    denom <- ncol(X)
  }
  G <- tcrossprod(centered) / denom
  ids <- as.character(seq_len(nrow(X)))
  rownames(G) <- ids
  colnames(G) <- ids
  G
}

extract_varcomp <- function(fit) {
  vc <- tryCatch(summary(fit)$varcomp, error = function(e) NULL)
  if (is.null(vc)) {
    stop("Could not extract variance components from sommer fit.")
  }
  vc_df <- as.data.frame(vc)
  vc_df$component <- rownames(vc)
  value_cols <- intersect(c("VarComp", "VarComp.", "VarComp..", "Variance", "Estimate"), colnames(vc_df))
  if (length(value_cols) == 0) {
    numeric_cols <- colnames(vc_df)[vapply(vc_df, is.numeric, logical(1))]
    if (length(numeric_cols) == 0) {
      stop("Could not find a numeric variance-component column.")
    }
    value_col <- numeric_cols[1]
  } else {
    value_col <- value_cols[1]
  }
  values <- as.numeric(vc_df[[value_col]])
  components <- as.character(vc_df$component)
  genetic_idx <- grep("id", components, ignore.case = TRUE)
  residual_idx <- grep("units|residual", components, ignore.case = TRUE)
  if (length(genetic_idx) == 0 || length(residual_idx) == 0) {
    if (length(values) < 2) {
      stop("Could not identify genetic and residual variance components.")
    }
    genetic_idx <- 1
    residual_idx <- length(values)
  } else {
    genetic_idx <- genetic_idx[1]
    residual_idx <- residual_idx[1]
  }
  list(
    genetic_variance = values[genetic_idx],
    residual_variance = values[residual_idx],
    value_column = value_col,
    varcomp_table = vc_df
  )
}

G <- build_g_matrix(X)
ids <- as.character(seq_len(nrow(X)))
df <- data.frame(id = factor(ids, levels = ids), y = y)

fit <- switch(
  sommer_method,
  "mmer" = sommer::mmer(y ~ 1, random = ~ sommer::vsr(id, Gu = G), rcov = ~ units, data = df, verbose = FALSE),
  "mmes" = sommer::mmes(
    y ~ 1,
    random = ~ sommer::vsm(sommer::ism(id), Gu = G),
    rcov = ~ units,
    data = df,
    verbose = FALSE
  ),
  stop(paste("Unsupported sommer method:", sommer_method))
)

vc <- extract_varcomp(fit)
vg <- as.numeric(vc$genetic_variance)
ve <- as.numeric(vc$residual_variance)
h2 <- vg / (vg + ve)

write_json(
  list(
    model = "GBLUP",
    sommer_method = sommer_method,
    n_samples = nrow(X),
    n_snps = ncol(X),
    genetic_variance = vg,
    residual_variance = ve,
    h2 = h2,
    varcomp_value_column = vc$value_column,
    varcomp_table = vc$varcomp_table
  ),
  path = out_path,
  auto_unbox = TRUE,
  pretty = TRUE,
  dataframe = "rows"
)
