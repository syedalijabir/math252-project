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



####################### 6. SELECT FINAL K #######################
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

####################### 7. FIT FINAL K-MEANS MODEL #######################
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


####################### 11. SAVE RESULTS #######################
dir.create("output", showWarnings = FALSE)
dir.create("output/embeddings", recursive = TRUE, showWarnings = FALSE)

write.csv(cluster_df, "output/kmeans_cluster_assignments.csv", row.names = FALSE)
write.csv(k_results_all, "output/kmeans_k_selection_metrics_all.csv", row.names = FALSE)
write.csv(k_results_mq, "output/kmeans_k_selection_metrics_macqueen.csv", row.names = FALSE)

saveRDS(kmeans_result, "output/kmeans_result.rds")
saveRDS(k_results_all, "output/k_results_all.rds")
saveRDS(k_results_mq, "output/k_results_mq.rds")


####################### 14. PLOT TOP GENRES #######################
ggplot(top_genres, aes(x = reorder(Genres, prop), y = prop, fill = as.factor(Cluster))) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  facet_wrap(~ Cluster, scales = "free_y") +
  theme_minimal() +
  labs(
    title = "Top 5 Genres by Cluster",
    x = "Genre",
    y = "Proportion within Cluster"
  )

ggplot(top_genres, aes(x = reorder(Genres, prop), y = prop, fill = as.factor(Cluster))) +
  geom_col(position = "dodge") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Top Genre Preferences by Cluster",
    x = "Genre",
    y = "Proportion within Cluster",
    fill = "Cluster"
  )


####################### NOTES FOR REPORTS #######################
# "The k-means results with k = 2 suggest that the embedding space contains one dominant user group and one smaller but distinct subgroup. The larger cluster represents the majority of users with more typical rating behavior, while the smaller cluster appears to capture users with distinct engagement patterns, rating tendencies, or genre preferences. Differences in activity level, average ratings, and genre proportions can be used to characterize the behavioral meaning of the two groups."