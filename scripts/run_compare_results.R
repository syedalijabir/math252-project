# 5) new file: `scripts/run_compare_results.R`


suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
})

source(file.path("Report", "R", "compare_results_helpers.R"))

results <- build_compare_results_data(
  cache_root = file.path("output", "benchmark_cache"),
  save_csv = TRUE,
  output_dir = file.path("output", "comparison_results")
)

dir.create(file.path("Report", "figures", "compare_results"), recursive = TRUE, showWarnings = FALSE)

ggsave(
  filename = file.path("Report", "figures", "compare_results", "best_silhouette_by_method_embedding.png"),
  plot = plot_best_silhouette_by_method_embedding(results$plot_df),
  width = 8.5,
  height = 5.2,
  dpi = 300
)

ggsave(
  filename = file.path("Report", "figures", "compare_results", "runtime_vs_silhouette.png"),
  plot = plot_runtime_vs_silhouette(results$plot_df),
  width = 8.5,
  height = 5.5,
  dpi = 300
)

ggsave(
  filename = file.path("Report", "figures", "compare_results", "average_ari_by_method_pair.png"),
  plot = plot_ari_method_pairs(results$ari_df),
  width = 8.5,
  height = 5.2,
  dpi = 300
)

ggsave(
  filename = file.path("Report", "figures", "compare_results", "silhouette_vs_k_kmeans_pd.png"),
  plot = plot_silhouette_vs_k(results$plot_df, methods = c("KMeans", "PD")),
  width = 10,
  height = 4.8,
  dpi = 300
)

readr::write_csv(results$best_table, file.path("output", "comparison_results", "best_models_report_table.csv"))
