####################### 1. SETUP #######################
source("setup.R")

####################### 2. CHOOSE NUMBER OF CLUSTERS ####################### 

results <- vector("list", length = max(k_vals))
avg_silhouette <- rep(NA, max(k_vals))
ch_scores <- rep(NA, max(k_vals))

for (k in k_vals) {
  cat("Running PDC for k =", k, "\n")
  
  # Run PD Clustering
  pdc_model <- PDC(embedding_matrix, k = k)
  
  # Extract labels
  labels <- pdc_model$label
  
  # Compute silhouette
  sil <- silhouette(labels, dmat) # distance matrix was computed in setup as dmat
  
  # Store average silhouette width
  avg_sil <- mean(sil[, 3])
  avg_silhouette[k] <- avg_sil
  
  ch_scores[k] <- intCriteria(
    traj = as.matrix(embedding_matrix),
    part = as.integer(labels),
    crit = "Calinski_Harabasz"
  )$calinski_harabasz
  
  # Save everything
  results[[k]] <- list(
    model = pdc_model,
    labels = labels,
    silhouette = sil,
    avg_silhouette = avg_sil,
    ch = ch_scores[k]
  )
  
  cat("Avg silhouette for k =", k, ":", avg_sil, "\n\n")
}

####################### 3. RESULTS TABLE #######################

pd_results <- data.frame(
  Method = "PD",
  k = k_vals,
  Silhouette = avg_silhouette[k_vals],
  CH = ch_scores[k_vals]
)

print(pd_results)



####################### 4. PLOT MODEL SELECTION #######################
plot(k_vals, avg_silhouette[k_vals],
     type = "b",
     pch = 19,
     col = "blue",
     xlab = "Number of Clusters (k)",
     ylab = "Average Silhouette Width",
     main = "PD-Clustering Model Selection")


####################### 5. CHOOSE BEST K #######################
best_k <- k_vals[which.max(avg_silhouette[k_vals])]
cat("Best k based on silhouette:", best_k, "\n")

####################### 6. INSPECT BEST SOLUTION #######################

best_model <- results[[best_k]]$model
best_labels <- results[[best_k]]$labels
best_sil <- results[[best_k]]$silhouette
final_avg_sil <- results[[best_k]]$avg_silhouette

# Silhouette plot
plot(
  best_sil,
  main = paste("PD Clustering Silhouette Plot for k =", best_k),
  col = "blue",
  border = NA
)

# Cluster sizes
cat("Cluster sizes for best k =", best_k, "\n")
print(table(best_labels))

# PD-specific probability silhouette
Silh(best_model$probability)


####################### 7. FINAL CLUSTER ASSIGNMENTS #######################
cluster_df <- data.frame(UserID = user_ids, Cluster = best_labels)

head(cluster_df)

####################### 8. PCA VISUALIZATION #######################

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
    title = paste("PD Clustering (PCA Projection), k =", best_k),
    x = paste0("PC1 (", round(100 * var_explained[1], 1), "%)"),
    y = paste0("PC2 (", round(100 * var_explained[2], 1), "%)")
  )

#### CLUSTER-SPECIFIC PCA PLOTS ####
for (cl in levels(pca_df$Cluster)) {
  print(
    ggplot(subset(pca_df, Cluster == cl), aes(x = PC1, y = PC2, color = Cluster)) +
      geom_point(alpha = 0.6) +
      theme_minimal() +
      ggtitle(paste("PD Cluster", cl, "(PCA Projection)"))
  )
}

labs(
  title = paste("PD Clustering (PCA Projection), k =", best_k),
  x = paste0("PC1 (", round(100 * var_explained[1], 1), "%)"),
  y = paste0("PC2 (", round(100 * var_explained[2], 1), "%)")
)

####################### 9. SAVE RESULTS #######################
dir.create("output", showWarnings = FALSE)

write.csv(cluster_df, "output/pd_cluster_assignments.csv", row.names = FALSE)
write.csv(pd_results, "output/pd_k_selection_metrics.csv", row.names = FALSE)

saveRDS(best_model, "output/pd_best_model.rds")
saveRDS(pd_results, "output/pd_results.rds")
saveRDS(cluster_df, "output/pd_cluster_df.rds")
saveRDS(pca_df, "output/pd_pca_df.rds")

