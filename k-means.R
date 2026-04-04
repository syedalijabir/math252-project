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

################### ADD TO REPORT ##########################
#* What each algorithm means:
#* 1. Hartigan–Wong (default) -- most theoretically refined
#* got Quick-Transfer warnings & struggles with this data --> not ideal here
#* 2. Lloyd -- Most commonly used in practice (usually in ML) --> Stable, simple
#* Warnings in run: "did not converge in 100 iterations" warnings
#* Good, but slightly unstable in this case
#* 3. MacQueen --> Online / incremental version; Simpler updates
#* Results: no warnings + consistent results
#* Best behavior for this data

#* INTERPRETATION FOR REPORT
#* Multiple k-means algorithms (Hartigan–Wong, Lloyd, and MacQueen) were evaluated to assess robustness. All algorithms produced consistent results, with k = 2 identified as the optimal number of clusters based on silhouette and Calinski–Harabasz indices. However, the Hartigan–Wong algorithm produced convergence warnings, and the Lloyd algorithm required additional iterations. The MacQueen algorithm exhibited stable behavior without convergence issues, and therefore was used for final clustering results.
#* The consistency of results across algorithms suggests that the observed clustering structure is inherent to the data rather than dependent on the optimization procedure.

#* What each column is telling us:
#* WSS (Within-Cluster Sum of Squares) -->Decreases as k increases (expected)
#* Measures compactness
#* ALWAYS goes down → not enough alone to choose k
#* WSS RESULTS:  Smooth decrease → no strong elbow

#* Silhouette (MOST IMPORTANT) --> Range: -1 to 1.
#* 0.3 = decent structure; ~0 = overlapping clusters; < 0 = bad clustering

#* SILHOUETTE RESULTS:
#* k=2 → 0.33   (GOOD); k=3 → 0.22 (OK); k≥4 → mostly negative (BAD)\
#* strong evidence.

#* CH (Calinski-Harabasz) --> Higher = better separation; Usually decreases as k increases
#* CH RESULTS:
#* k=2 → 139  (highest); k=3 → 113; k=4 → 91; ...
#* Also favors k = 2

#* REPORT
#* The optimal number of clusters was selected based on multiple internal validation metrics, including silhouette score and Calinski–Harabasz index. The silhouette score was highest at k = 2 (0.33), indicating strong cluster separation. The Calinski–Harabasz index also peaked at k = 2 (139), further supporting this choice. For k ≥ 4, silhouette scores became negative, indicating poor clustering structure and overlapping clusters. Therefore, k = 2 was selected as the optimal number of clusters.
#* Why does this happen? The Embeddings likely capture a broad split in user behavior and not fine-grained segmentation
#* 2 clusters = meaningful structure; more clusters = forced splits → bad separation
#* 
#* Although k = 5 yields a positive silhouette score (0.13), it is substantially lower than the value at k = 2 (0.33), indicating weaker cluster separation. Additionally, the Calinski–Harabasz index decreases significantly as k increases. This suggests that increasing the number of clusters introduces artificial fragmentation rather than meaningful structure. Therefore, k = 2 is selected as the optimal number of clusters.
#* When k = 10, the resulting clusters are highly imbalanced, with two dominant clusters containing over 1700 users each, while several clusters contain fewer than 200 users. This imbalance suggests that the algorithm is artificially subdividing a smaller number of underlying groups. This observation is consistent with the silhouette analysis, which indicated that k = 2 provides the best clustering structure. Larger values of k introduce unstable and weakly defined clusters.
#* The presence of a few large clusters alongside several very small clusters indicates that the underlying data structure is not well-separated into many groups. Instead, the data appears to exhibit a coarse partitioning into a small number of dominant behavioral patterns.
###############################################

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
write.csv(rating_summary, "output/kmeans_rating_summary.csv", row.names = FALSE)
write.csv(activity_summary, "output/kmeans_activity_summary.csv", row.names = FALSE)
write.csv(genre_summary, "output/kmeans_genre_summary.csv", row.names = FALSE)

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