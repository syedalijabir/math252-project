####################### 1. SETUP #######################
source("setup.R")

####################### 2. CHOOSE NUMBER OF CLUSTERS ####################### 
# Run all three algorithms
k_results_hw <- evaluate_kmeans(embedding_matrix, k_vals, dmat, "Hartigan-Wong")
k_results_lloyd <- evaluate_kmeans(embedding_matrix, k_vals, dmat, "Lloyd")
k_results_mq <- evaluate_kmeans(embedding_matrix, k_vals, dmat, "MacQueen")

# Combine results into one table
k_results_all <- dplyr::bind_rows(k_results_hw, k_results_lloyd, k_results_mq)

# Comparison Table for all results (wide to be easier to compare)
k_results_wide <- k_results_all %>% 
  pivot_wider(names_from = Algorithm, values_from = c(WSS, Silhouette, CH))

print(k_results_wide)


####################### 3. PLOT K-SELECTION METRICS #######################
# Comparison plots
ggplot(k_results_all, aes(x = k, y = Silhouette, color = Algorithm)) +
  geom_line() + geom_point() + theme_minimal() + 
  labs(title = "Silhouette by k and k-means algorithm")

ggplot(k_results_all, aes(x = k, y = CH, color = Algorithm)) +
  geom_line() + geom_point() + theme_minimal() +
  labs(title = "Calinski-Harabasz by k and k-means algorithm")

ggplot(k_results_all, aes(x = k, y = WSS, color = Algorithm)) +
  geom_line() + geom_point() + theme_minimal() +
  labs(title = "WSS by k and k-means algorithm")

# JUST MACQUEEN --> Use MacQueen for final results, Show all algorithms agree --> Emphasize robustness, not superiority
print(k_results_mq)


ggplot(k_results_mq, aes(x = k, y = Silhouette, color = Algorithm)) +
  geom_line() + geom_point() + theme_minimal() + 
  labs(title = "Silhouette by k and k-means algorithm")

ggplot(k_results_mq, aes(x = k, y = CH, color = Algorithm)) +
  geom_line() + geom_point() + theme_minimal() +
  labs(title = "Calinski-Harabasz by k and k-means algorithm")

ggplot(k_results_mq, aes(x = k, y = WSS, color = Algorithm)) +
  geom_line() + geom_point() + theme_minimal() +
  labs(title = "WSS by k and k-means algorithm")

####################### 4. SELECT FINAL K #######################
#* Choose final k from MacQueen results -- Show all algorithms agree --> Emphasize robustness, not superiority

#* choose k using silhouette as primary criterion
#* use CH as confirmation
#* check cluster size balance before finalizing
#* set k manually after inspecting results.

# Choose final k from MacQueen results
best_k_sil <- k_results_mq$k[which.max(k_results_mq$Silhouette)]
best_k_ch  <- k_results_mq$k[which.max(k_results_mq$CH)]

cat("Best k by silhouette:", best_k_sil, "\n", "Best k by CH:", best_k_ch, "\n")

final_k <- best_k_sil

####################### 5. FIT FINAL K-MEANS MODEL #######################
set.seed(123)

kmeans_result <- kmeans(
  embedding_matrix,
  centers = final_k,
  nstart = 100, 
  iter.max = 100,
  algorithm = "MacQueen"
)

# Save cluster labels with user IDs
cluster_df <- data.frame(
  UserID = user_ids,
  Cluster = kmeans_result$cluster
)

# Check cluster sizes 
table(kmeans_result$cluster)

####################### 9. FINAL SILHOUETTE SCORE #######################
final_sil <- silhouette(kmeans_result$cluster, dmat)
final_avg_sil <- mean(final_sil[, 3])

cat("Final average silhouette width:", final_avg_sil, "\n")

# Silhouette plot
plot(
  final_sil,
  main = paste("Silhouette Plot for k =", final_k),
  col = "blue",
  border = NA
)



####################### 10. PCA VISUALIZATION #######################
# For visualization, not for fitting k-means.
pca <- prcomp(embedding_matrix)

var_explained <- summary(pca)$importance[2, 1:2]

pca_df <- data.frame(
  PC1 = pca$x[, 1],
  PC2 = pca$x[, 2],
  Cluster = as.factor(kmeans_result$cluster)
)

ggplot(pca_df, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(alpha = 0.6) +
  theme_minimal() +
  labs(
    title = paste("K-Means Clusters (PCA Projection), k =", final_k),
    x = paste0("PC1 (", round(100 * var_explained[1], 1), "%)"),
    y = paste0("PC2 (", round(100 * var_explained[2], 1), "%)")
  )



####################### 11. SAVE RESULTS #######################

write.csv(cluster_df, "output/kmeans_cluster_assignments.csv", row.names = FALSE)
write.csv(k_results_all, "output/kmeans_k_selection_metrics_all.csv", row.names = FALSE)
write.csv(k_results_mq, "output/kmeans_k_selection_metrics_macqueen.csv", row.names = FALSE)

saveRDS(kmeans_result, "output/kmeans_result.rds")
saveRDS(k_results_all, "output/k_results_all.rds")
saveRDS(k_results_mq, "output/k_results_mq.rds")




