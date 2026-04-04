####################### 1. SETUP #######################
source("setup.R")

# Required package for Factorial PD Clustering
if (!requireNamespace("FPDclustering", quietly = TRUE)) {
  install.packages("FPDclustering")
}
library(FPDclustering)

####################### 2. LOAD EMBEDDINGS #######################
# Change this path to 128 or 256 for final runs if needed

embedding_path <- "output/embeddings/user_embeddings_64.csv"

embeddings <- read_csv(embedding_path, show_col_types = FALSE)

user_ids <- embeddings[[1]]
embedding_matrix <- embeddings[, -1]

# Scale embeddings before clustering
embedding_matrix <- scale(as.matrix(embedding_matrix))

cat("Embedding matrix dimensions:", dim(embedding_matrix), "\n")

####################### 3. OPTIONAL TUCKER FACTOR CHECK #######################
# This can be used as a rough guide for choosing q

TuckerFactors(embedding_matrix, 6)

####################### 4. TUNE CLUSTERS AND FACTOR DIMENSIONS #######################
# FPDC requires:
# - k = number of clusters
# - q = factor dimension
# - maxiter = maximum number of iterations

# FINAL GRID
k_range <- 2:10
q_range <- 2:6

# DEVELOPMENT GRID (uncomment for testing only)
# k_range <- 2:4
# q_range <- 2:3

x_mat <- as.matrix(embedding_matrix)

####################### 5. OPTIONAL SPEEDUP: SAMPLED SILHOUETTE #######################
# Use sampled silhouette during tuning for speed.
# Compute full silhouette only for the final selected model.

use_sampled_silhouette <- TRUE
sil_sample_size <- 1500

if (use_sampled_silhouette) {
  set.seed(42)
  sil_idx <- sort(sample(seq_len(nrow(x_mat)), min(sil_sample_size, nrow(x_mat))))
  dmat_sub <- dist(x_mat[sil_idx, , drop = FALSE])
} else {
  dmat_full <- dist(x_mat)
}

####################### 6. PARALLEL GRID SEARCH #######################
if (!requireNamespace("future.apply", quietly = TRUE)) {
  install.packages("future.apply")
}
library(future.apply)

plan(multisession)

grid <- expand.grid(k = k_range, q = q_range)

fpdc_results_list <- future_lapply(seq_len(nrow(grid)), function(i) {
  k <- grid$k[i]
  q <- grid$q[i]
  
  cat("Running FPDC with k =", k, "and q =", q, "\n")
  
  model <- FPDC(
    x_mat,
    k,
    50,   # max iterations
    q
  )
  
  labels <- model$label
  
  # Silhouette
  if (use_sampled_silhouette) {
    sil <- silhouette(labels[sil_idx], dmat_sub)
  } else {
    sil <- silhouette(labels, dmat_full)
  }
  avg_sil <- mean(sil[, 3])
  
  # Calinski-Harabasz
  ch_val <- intCriteria(
    traj = x_mat,
    part = as.integer(labels),
    crit = "Calinski_Harabasz"
  )$calinski_harabasz
  
  list(
    k = k,
    q = q,
    result = data.frame(
      Method = "FPDC",
      k = k,
      q = q,
      Silhouette = avg_sil,
      CH = ch_val
    ),
    model = model,
    labels = labels
  )
})

####################### 7. RESULTS TABLE #######################

fpdc_results <- dplyr::bind_rows(lapply(fpdc_results_list, `[[`, "result")) %>%
  arrange(desc(Silhouette), desc(CH))

print(fpdc_results)

####################### 8. PLOT MODEL SELECTION #######################

ggplot(fpdc_results, aes(x = q, y = Silhouette, color = factor(k), group = k)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  labs(
    title = "FPDC Model Selection by Silhouette",
    x = "Factor Dimension (q)",
    y = "Average Silhouette Width",
    color = "Clusters (k)"
  )

ggplot(fpdc_results, aes(x = q, y = CH, color = factor(k), group = k)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  labs(
    title = "FPDC Model Selection by Calinski-Harabasz",
    x = "Factor Dimension (q)",
    y = "CH Index",
    color = "Clusters (k)"
  )

####################### 9. CHOOSE BEST MODEL #######################
# Primary criterion: silhouette
# Secondary criterion: CH

best_index <- which.max(sapply(fpdc_results_list, function(x) x$result$Silhouette))
best_model_obj <- fpdc_results_list[[best_index]]

best_k <- best_model_obj$result$k
best_q <- best_model_obj$result$q

cat("Best FPDC model based on tuning:\n")
print(best_model_obj$result)

best_model <- best_model_obj$model
best_labels <- best_model_obj$labels
final_avg_sil_tuning <- best_model_obj$result$Silhouette
final_ch <- best_model_obj$result$CH

####################### 10. FULL SILHOUETTE FOR FINAL MODEL #######################
# Even if tuning used sampled silhouette, compute full silhouette here for final reporting

dmat_full <- dist(x_mat)
best_sil <- silhouette(best_labels, dmat_full)
final_avg_sil_full <- mean(best_sil[, 3])

plot(
  best_sil,
  main = paste("FPDC Silhouette Plot for k =", best_k, ", q =", best_q),
  col = "blue",
  border = NA
)

cat("Cluster sizes for best FPDC model (k =", best_k, ", q =", best_q, ")\n")
print(table(best_labels))

cat("Tuning silhouette:", final_avg_sil_tuning, "\n")
cat("Final full-data silhouette:", final_avg_sil_full, "\n")
cat("Final CH index:", final_ch, "\n")

####################### 11. PROBABILITY-BASED SILHOUETTE #######################
# FPDC-specific membership quality diagnostic

Silh(best_model$probability)

####################### 12. FINAL CLUSTER ASSIGNMENTS #######################

cluster_df <- data.frame(
  UserID = user_ids,
  Cluster = best_labels
)

head(cluster_df)

####################### 13. PCA VISUALIZATION #######################

pca <- prcomp(x_mat)
var_explained <- summary(pca)$importance[2, 1:2]

pca_df <- data.frame(
  PC1 = pca$x[, 1],
  PC2 = pca$x[, 2],
  Cluster = as.factor(best_labels)
)

ggplot(pca_df, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(alpha = 0.6) +
  theme_minimal() +
  labs(
    title = paste("Factorial PD Clustering (PCA Projection), k =", best_k, ", q =", best_q),
    x = paste0("PC1 (", round(100 * var_explained[1], 1), "%)"),
    y = paste0("PC2 (", round(100 * var_explained[2], 1), "%)")
  )

####################### 14. SAVE RESULTS #######################

dir.create("output", showWarnings = FALSE)

write.csv(cluster_df, "output/fpdc_cluster_assignments.csv", row.names = FALSE)
write.csv(fpdc_results, "output/fpdc_model_selection_metrics.csv", row.names = FALSE)

saveRDS(best_model, "output/fpdc_best_model.rds")
saveRDS(fpdc_results, "output/fpdc_results.rds")
saveRDS(cluster_df, "output/fpdc_cluster_df.rds")
saveRDS(pca_df, "output/fpdc_pca_df.rds")
saveRDS(var_explained, "output/fpdc_var_explained.rds")