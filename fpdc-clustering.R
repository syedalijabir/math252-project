####################### 1. SETUP #######################
source("setup.R")

####################### 2. LOAD EMBEDDINGS #######################
embedding_path <- "output/embeddings/user_embeddings_64.csv"

embeddings <- read_csv(embedding_path, show_col_types = FALSE)

user_ids <- embeddings[[1]]
embedding_matrix <- scale(as.matrix(embeddings[, -1]))

cat("Embedding matrix dimensions:", dim(embedding_matrix), "\n")

####################### 3. TUCKER CHECK #######################
# Skip this if it is too slow
TuckerFactors(embedding_matrix, 3)

####################### 4. TUNE k AND q #######################
# Keep this grid small for full-data FPDC

k_range <- 2:4
q_range <- 2:3

dmat <- dist(embedding_matrix)

results_list <- list()
best_model <- NULL
best_labels <- NULL
best_k <- NA
best_q <- NA
best_sil <- -Inf
best_ch <- NA
best_prob_sil <- NA
best_sil_obj <- NULL

counter <- 1

for (k in k_range) {
  for (q in q_range) {
    cat("Running FPDC with k =", k, "and q =", q, "\n")
    
    model <- FPDC(
      embedding_matrix,
      k,
      20,   # reduced iterations for tuning
      q
    )
    
    labels <- model$label
    
    sil <- silhouette(labels, dmat)
    avg_sil <- mean(sil[, 3])
    
    ch_val <- intCriteria(
      traj = embedding_matrix,
      part = as.integer(labels),
      crit = "Calinski_Harabasz"
    )$calinski_harabasz
    
    prob_sil <- Silh(model$probability)
    
    results_list[[counter]] <- data.frame(
      Method = "FPDC",
      k = k,
      q = q,
      Silhouette = avg_sil,
      CH = ch_val,
      ProbSilh = prob_sil
    )
    
    if (avg_sil > best_sil) {
      best_sil <- avg_sil
      best_ch <- ch_val
      best_prob_sil <- prob_sil
      best_k <- k
      best_q <- q
      best_model <- model
      best_labels <- labels
      best_sil_obj <- sil
    }
    
    counter <- counter + 1
  }
}

####################### 5. RESULTS TABLE #######################

fpdc_results <- bind_rows(results_list) %>%
  arrange(desc(Silhouette), desc(CH))

print(fpdc_results)

####################### 6. PLOT MODEL SELECTION #######################

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

ggplot(fpdc_results, aes(x = q, y = ProbSilh, color = factor(k), group = k)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  labs(
    title = "FPDC Model Selection by Probability Silhouette",
    x = "Factor Dimension (q)",
    y = "Probability-Based Silhouette",
    color = "Clusters (k)"
  )

####################### 7. REFIT BEST MODEL #######################
# Refit best configuration with more iterations

cat("Best FPDC tuning configuration:\n")
cat("k =", best_k, "\n")
cat("q =", best_q, "\n")
cat("Silhouette =", best_sil, "\n")
cat("CH =", best_ch, "\n")
cat("Probability silhouette =", best_prob_sil, "\n")

best_model <- FPDC(
  embedding_matrix,
  best_k,
  50,   # more iterations for final fit
  best_q
)

best_labels <- best_model$label

####################### 8. FINAL EVALUATION #######################

best_sil_obj <- silhouette(best_labels, dmat)
final_avg_sil <- mean(best_sil_obj[, 3])

final_ch <- intCriteria(
  traj = embedding_matrix,
  part = as.integer(best_labels),
  crit = "Calinski_Harabasz"
)$calinski_harabasz

final_prob_sil <- Silh(best_model$probability)

plot(
  best_sil_obj,
  main = paste("FPDC Silhouette Plot, k =", best_k, ", q =", best_q),
  col = "blue",
  border = NA
)

cat("Cluster sizes for best FPDC model:\n")
print(table(best_labels))

cat("Final silhouette:", final_avg_sil, "\n")
cat("Final CH index:", final_ch, "\n")
cat("Final probability silhouette:", final_prob_sil, "\n")

####################### 9. FINAL CLUSTER ASSIGNMENTS #######################

cluster_df <- data.frame(
  UserID = user_ids,
  Cluster = best_labels
)

head(cluster_df)

####################### 10. PCA VISUALIZATION #######################

pca <- prcomp(embedding_matrix)
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
    title = paste("FPDC (PCA Projection), k =", best_k, ", q =", best_q),
    x = paste0("PC1 (", round(100 * var_explained[1], 1), "%)"),
    y = paste0("PC2 (", round(100 * var_explained[2], 1), "%)")
  )

####################### 11. SAVE RESULTS #######################

dir.create("output", showWarnings = FALSE)

write.csv(cluster_df, "output/fpdc_cluster_assignments.csv", row.names = FALSE)
write.csv(fpdc_results, "output/fpdc_model_selection_metrics.csv", row.names = FALSE)

saveRDS(best_model, "output/fpdc_best_model.rds")
saveRDS(fpdc_results, "output/fpdc_results.rds")
saveRDS(cluster_df, "output/fpdc_cluster_df.rds")
saveRDS(pca_df, "output/fpdc_pca_df.rds")
saveRDS(var_explained, "output/fpdc_var_explained.rds")
