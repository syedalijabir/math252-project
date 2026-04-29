# =============================================================================
# NU SWEEP FOR FPDC  —  Standalone Script
# =============================================================================
#
# GOAL:
#   Find a good value of the `nu` parameter for FPDC (Factorial PD Clustering)
#   before committing to a full expensive grid search over k and q.
#
# WHAT IS nu?
#   In FPDC, `nu` controls the degrees of freedom of the t-distribution used
#   to model cluster membership probabilities:
#     - Low nu  (e.g. 1-5):  heavy tails → softer, fuzzier cluster assignments.
#                             More robust to outliers but can blur boundaries.
#     - High nu (e.g. 50+):  approaches a Gaussian → sharper, harder assignments.
#                             More sensitive to outliers.
#   A good nu produces well-separated clusters as measured by Silhouette,
#   Calinski-Harabasz (CH), and Probability Silhouette.
#
# STRATEGY:
#   To keep runtime short, this sweep:
#     1. Uses only the 64-dim embedding (smallest / fastest).
#     2. Fixes k and q at mid-range values (not tuning them here).
#     3. Runs on a random SUBSAMPLE of users rather than the full dataset.
#   Once a good nu is identified, plug it into the main grid search loop
#   (the run-fpdc chunk in factorial-pd-clustering.Rmd) which uses the full
#   data and sweeps k and q.
#
# CACHING:
#   Each (nu, k, q, subsample_size) combination is saved to disk.
#   Re-running this script will load from cache instantly.
#
# HOW TO USE:
#   1. Set the configuration variables below (paths, nu values, k, q, n).
#   2. Run the script: Rscript nu_sweep.R
#      or source it inside RStudio: source("nu_sweep.R")
#   3. Check the printed recommendation at the end and update nu in the
#      main Rmd accordingly.
# =============================================================================


# ── Libraries ─────────────────────────────────────────────────────────────────

library(readr)          # read_csv
library(tidyr)          # pivot_longer for plotting
library(ggplot2)        # plots
library(cluster)        # silhouette()
library(clusterCrit)    # intCriteria() for Calinski-Harabasz
library(FPDclustering)  # FPDC(), Silh()


# ── Configuration ─────────────────────────────────────────────────────────────

# Path to the embedding file to use for this sweep.
# Using 64-dim only — smallest and fastest of the three embeddings.
nu_sweep_embedding <- "output/embeddings/user_embeddings_64.csv"

# nu values to test — log-spaced to cover a wide range efficiently.
# Covers soft (nu=1) through near-Gaussian (nu=100) behavior.
nu_values <- c(5, 10, 15, 20, 30, 50)

# Fix k and nf at mid-range values from the main grid (k: 2-8).
# nf = number of factors for variables (what the main Rmd calls q).
# The fourth argument to FPDC() is nu — that is what we are sweeping.
# Signature: FPDC(data, k, nf, nu)
nu_fixed_k  <- 4
nu_fixed_nf <- 20   # matches the hardcoded nf=20 in the main Rmd

# Subsample size: number of users to draw for each FPDC call.
# 1000-2000 is usually enough to get a reliable signal for nu without
# the full-data runtime cost.
nu_subsample_n <- 3000

# Random seed — ensures the same subsample is drawn every time,
# so results are reproducible and cache hits work correctly.
nu_sweep_seed <- 42

# Output directories — mirrors the structure used in the main Rmd.
dir.create("output/fpdc_cache/nu_sweep", recursive = TRUE, showWarnings = FALSE)
dir.create("output/fpdc_plots",          recursive = TRUE, showWarnings = FALSE)


# ── Load data ─────────────────────────────────────────────────────────────────

nu_tag <- tools::file_path_sans_ext(basename(nu_sweep_embedding))

if (!file.exists(nu_sweep_embedding)) {
  stop("Embedding file not found: ", nu_sweep_embedding,
       "\nCheck that nu_sweep_embedding points to the correct path.")
}

cat("Loading embedding from disk:", nu_sweep_embedding, "\n")
nu_raw          <- read_csv(nu_sweep_embedding, show_col_types = FALSE)
nu_userids_full <- nu_raw[[1]]                      # first column = user IDs
nu_matrix_full  <- scale(as.matrix(nu_raw[, -1]))   # remaining columns = embeddings, scaled

cat(sprintf("Full dataset: %d users x %d dims\n",
            nrow(nu_matrix_full), ncol(nu_matrix_full)))


# ── Subsample ─────────────────────────────────────────────────────────────────

# Draw a random subset of rows for faster iteration.
# The cache filename includes the subsample size, so if you later switch to
# the full dataset the cached files won't collide with these.

set.seed(nu_sweep_seed)
nu_subsample_idx <- sample(nrow(nu_matrix_full), size = nu_subsample_n)
nu_matrix        <- nu_matrix_full[nu_subsample_idx, ]   # SUBSAMPLED matrix
nu_dmat          <- dist(nu_matrix)                       # SUBSAMPLED distance matrix

# ── To use the FULL dataset instead, comment out the 3 lines above and
#    uncomment the 2 lines below:
# nu_matrix <- nu_matrix_full
# nu_dmat   <- dist(nu_matrix_full)

cat(sprintf("Running nu sweep on subsample: %d users (seed = %d)\n",
            nrow(nu_matrix), nu_sweep_seed))


# ── Run sweep ─────────────────────────────────────────────────────────────────

# Results table — one row per nu value
nu_results <- data.frame(
  nu          = nu_values,
  Silhouette  = NA_real_,
  CH          = NA_real_,
  ProbSilh    = NA_real_,
  runtime_sec = NA_real_,
  status      = "ok",
  stringsAsFactors = FALSE
)

cat(sprintf("\nStarting nu sweep: %d values to test (k=%d, nf=%d, n=%d)\n",
            length(nu_values), nu_fixed_k, nu_fixed_nf, nrow(nu_matrix)))
cat(rep("-", 60), "\n", sep = "")

for (i in seq_along(nu_values)) {

  nu_val <- nu_values[i]

  # Cache filename includes subsample size so full-data and subsampled results
  # are stored separately and never overwrite each other
  cache_file <- file.path(
    "output/fpdc_cache/nu_sweep",
    sprintf("fpdc_%s_k%d_nf%d_nu%g_n%d.rds",
            nu_tag, nu_fixed_k, nu_fixed_nf, nu_val, nrow(nu_matrix))
  )

  # ── Load from cache if available ──────────────────────────────────────────
  if (file.exists(cache_file)) {
    cat(sprintf("[%d/%d] nu = %g  →  loading from cache\n",
                i, length(nu_values), nu_val))
    cached <- readRDS(cache_file)
    nu_results$Silhouette[i]  <- cached$Silhouette
    nu_results$CH[i]          <- cached$CH
    nu_results$ProbSilh[i]    <- cached$ProbSilh
    nu_results$runtime_sec[i] <- cached$runtime_sec
    next
  }

  # ── Run FPDC ──────────────────────────────────────────────────────────────
  cat(sprintf("[%d/%d] nu = %g  →  running FPDC ...\n",
              i, length(nu_values), nu_val))
  t0 <- Sys.time()

  # withCallingHandlers lets warnings be logged without interrupting execution
  # (unlike tryCatch, which stops the function when it catches a warning).
  # tryCatch wraps the outer call to catch hard errors only.
  model <- tryCatch({
    withCallingHandlers(
      FPDC(nu_matrix, nu_fixed_k, nu_fixed_nf, nu_val),
      warning = function(w) {
        nu_results$status[i] <<- "warning"
        cat(sprintf("         WARNING: %s\n", conditionMessage(w)))
        invokeRestart("muffleWarning")   # suppress but continue running
      }
    )
  },
  error = function(e) {
    nu_results$status[i] <<- "error"
    cat(sprintf("         ERROR: %s\n", conditionMessage(e)))
    NULL
  })

  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  nu_results$runtime_sec[i] <- elapsed

  if (is.null(model)) {
    cat(sprintf("         FAILED — skipping nu = %g\n", nu_val))
    next
  }

  # ── Compute metrics ───────────────────────────────────────────────────────
  labels   <- model$label
  avg_sil  <- mean(silhouette(labels, nu_dmat)[, 3])
  ch_val   <- intCriteria(nu_matrix, as.integer(labels),
                          "Calinski_Harabasz")$calinski_harabasz
  prob_sil <- Silh(model$probability)

  nu_results$Silhouette[i] <- avg_sil
  nu_results$CH[i]         <- ch_val
  nu_results$ProbSilh[i]   <- prob_sil

  cat(sprintf("         Done in %5.1f s  |  Silh = %6.4f  |  CH = %8.1f  |  ProbSilh = %6.4f\n",
              elapsed, avg_sil, ch_val, prob_sil))

  # ── Cache result ──────────────────────────────────────────────────────────
  saveRDS(
    list(
      model       = model,
      Silhouette  = avg_sil,
      CH          = ch_val,
      ProbSilh    = prob_sil,
      runtime_sec = elapsed,
      nu          = nu_val,
      k           = nu_fixed_k,
      nf          = nu_fixed_nf,
      n           = nrow(nu_matrix)
    ),
    cache_file
  )
}

cat(rep("-", 60), "\n", sep = "")
cat("Nu sweep complete. Summary:\n\n")
print(nu_results)


# ── Plot ───────────────────────────────────────────────────────────────────────

# Reshape to long format so we can facet by metric
nu_long <- pivot_longer(
  nu_results,
  cols      = c(Silhouette, CH, ProbSilh),
  names_to  = "metric",
  values_to = "value"
)

p_nu <- ggplot(nu_long, aes(x = nu, y = value)) +
  geom_line() +
  geom_point(size = 2) +
  facet_wrap(~ metric, scales = "free_y") +  # each metric on its own y-axis
  scale_x_log10(breaks = nu_values) +         # log scale since nu_values are log-spaced
  theme_minimal() +
  labs(
    title    = sprintf("FPDC nu sweep  |  k=%d, nf=%d, n=%d users  |  %s",
                       nu_fixed_k, nu_fixed_nf, nrow(nu_matrix), nu_tag),
    subtitle = "Higher Silhouette, CH, and ProbSilh = better clustering",
    x        = "nu (log scale)",
    y        = "Metric value"
  )

print(p_nu)

plot_path <- "output/fpdc_plots/nu_sweep.png"
ggsave(plot_path, p_nu, width = 10, height = 4, dpi = 120)
cat("\nPlot saved to:", plot_path, "\n")


# ── Recommendation ─────────────────────────────────────────────────────────────

# Rank each nu by each metric separately, then average the ranks.
# This avoids having to pick one primary metric and balances all three.
nu_ok <- nu_results[nu_results$status == "ok" & !is.na(nu_results$Silhouette), ]

if (nrow(nu_ok) > 0) {
  nu_ok$rank_sil  <- rank(-nu_ok$Silhouette)  # negative = higher is better
  nu_ok$rank_ch   <- rank(-nu_ok$CH)
  nu_ok$rank_prob <- rank(-nu_ok$ProbSilh)
  nu_ok$avg_rank  <- (nu_ok$rank_sil + nu_ok$rank_ch + nu_ok$rank_prob) / 3

  best_nu_row <- nu_ok[which.min(nu_ok$avg_rank), ]

  cat(sprintf(
    "\nRecommended nu = %g  (best average rank across Silhouette, CH, ProbSilh)\n",
    best_nu_row$nu
  ))
  cat("\nTo use this in the main Rmd (factorial-pd-clustering.Rmd),\n")
  cat("find the FPDC() call in the run-fpdc chunk and change:\n")
  cat("    nu = 10\n")
  cat(sprintf("to:\n    nu = %g\n", best_nu_row$nu))
} else {
  cat("\nNo successful runs to rank — check errors above.\n")
}
