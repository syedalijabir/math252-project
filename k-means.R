
####################### 1. REQUIRED PACKAGES/LIBRARIES  ####################### 
required_packages <- c(
  "readr",       # reading files 
  "dplyr",       # data manipulation
  "tidyr",       # splitting genres into rows
  "ggplot2",     # plotting
  "cluster",     # silhouette
  "clusterCrit", # Calinski-Harabasz index
  "factoextra",   # helper plots
  "FPDclustering",
  "hopkins", #############***
)

installed <- rownames(installed.packages())
missing_pkgs <- setdiff(required_packages, installed)
if (length(missing_pkgs) > 0) {
  install.packages(missing_pkgs)
}

library(readr); library(dplyr); library(tidyr); library(ggplot2)
library(cluster); library(clusterCrit); library(factoextra); 
library(FPDclustering); library(hopkins)

set.seed(42)


####################### 2. Load embeddings files ####################### 

embedding_path <- "output/embeddings/user_embeddings_128.csv"
embeddings <- read_csv(embedding_path, show_col_types = FALSE)
user_ids <- embeddings[[1]]          # Save user IDs, first column is UserID
embedding_matrix <- embeddings[, -1] # Remove UserID column. Remaining columns are embedding features

embedding_matrix <- scale(as.matrix(embedding_matrix)) # Convert to matrix and scale before clustering

cat("Embedding matrix dimensions:", dim(embedding_matrix), "\n")


####################### 4. CHOOSE NUMBER OF CLUSTERS #######################
#* We evaluate k = 2 to 20 using:
#* WSS (elbow method)
#* average silhouette width
#* Calinski-Harabasz (CH) index

####################### 4. CHOOSE NUMBER OF CLUSTERS #######################
# Evaluate k = 2 to 20 using:
# - WSS (elbow method)
# - average silhouette width
# - Calinski-Harabasz (CH) index

k_vals <- 2:20

# Compute distance matrix once for silhouette
dmat <- dist(embedding_matrix)

# Function to evaluate one k-means algorithm across all k
evaluate_kmeans <- function(x, k_vals, dmat, algorithm = "Hartigan-Wong",
                            nstart = 10, iter.max = 100) {
  
  wss <- numeric(length(k_vals))
  sil_scores <- numeric(length(k_vals))
  ch_scores <- numeric(length(k_vals))
  
  for (i in seq_along(k_vals)) {
    k <- k_vals[i]
    
    km <- kmeans(
      x,
      centers = k,
      nstart = nstart,
      iter.max = iter.max,
      algorithm = algorithm
    )
    
    wss[i] <- km$tot.withinss
    
    sil <- silhouette(km$cluster, dmat)
    sil_scores[i] <- mean(sil[, 3])
    
    ch_scores[i] <- intCriteria(
      traj = as.matrix(x),
      part = as.integer(km$cluster),
      crit = "Calinski_Harabasz"
    )$calinski_harabasz
  }
  
  data.frame(
    Algorithm = algorithm,
    k = k_vals,
    WSS = wss,
    Silhouette = sil_scores,
    CH = ch_scores
  )
}

# Run all three algorithms
k_results_hw <- evaluate_kmeans(x = embedding_matrix, k_vals = k_vals,
                                dmat = dmat, algorithm = "Hartigan-Wong"
)

k_results_lloyd <- evaluate_kmeans(x = embedding_matrix, k_vals = k_vals,
                                   dmat = dmat, algorithm = "Lloyd"
)

k_results_mq <- evaluate_kmeans(x = embedding_matrix, k_vals = k_vals, 
                                dmat = dmat, algorithm = "MacQueen"
)

# Combine all results into one table
k_results_all <- dplyr::bind_rows(k_results_hw, k_results_lloyd, k_results_mq)

# print combined results
print(k_results_all)


#### Comparison Table
k_results_wide <- k_results_all %>% 
  pivot_wider(names_from = Algorithm, values_from = c(WSS, Silhouette, CH))

print(k_results_wide)

#* What each algorithm means:
#* 1. Hartigan–Wong (default) -- most theoretically refined
#* got Quick-Transfer warnings & struggles with this data --> not ideal here
#* 2. Lloyd -- Most commonly used in practice (usually in ML) --> Stable, simple
#* Warnings in run: "did not converge in 100 iterations" warnings
#* Good, but slightly unstable in this case
#* 3. MacQueen --> Online / incremental version; Simpler updates
#* Results: no warnings + consistent results
#* Best behavior for this data

#* INTERPRETATION
#* Multiple k-means algorithms (Hartigan–Wong, Lloyd, and MacQueen) were evaluated to assess robustness. All algorithms produced consistent results, with k = 2 identified as the optimal number of clusters based on silhouette and Calinski–Harabasz indices. However, the Hartigan–Wong algorithm produced convergence warnings, and the Lloyd algorithm required additional iterations. The MacQueen algorithm exhibited stable behavior without convergence issues, and therefore was used for final clustering results.

###########################################################

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

####################### 5. PLOT K-SELECTION METRICS #######################
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

# JUST MCQUEEN
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

####################### 6. SELECT FINAL K #######################

#* choose k using silhouette as primary criterion
#* use CH as confirmation
#* check cluster size balance before finalizing
#* set k manually after inspecting results.

best_k_sil <- k_vals[which.max(sil_scores)]
best_k_ch <- k_vals[which.max(ch_scores)]

cat("Best k by silhouette:", best_k_sil, "\n")
cat("Best k by CH:", best_k_ch, "\n")

# Set final k here after reviewing plots and metrics.
# From results, k = 2 looked strongest.
final_k <- 2


####################### 7. FIT FINAL K-MEANS MODEL #######################

set.seed(123)

kmeans_result <- kmeans(
  embedding_matrix,
  centers = final_k,
  nstart = 100, 
  algorithm = "MacQueen"
)

# Save cluster labels with user IDs
cluster_df <- data.frame(
  UserID = user_ids,
  Cluster = kmeans_result$cluster
)

head(cluster_df)


####################### 8. CHECK CLUSTER SIZES #######################

cat("Cluster sizes for k =", final_k, "\n")
print(table(kmeans_result$cluster))


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
# This is just for visualization, not for fitting k-means.

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


####################### 11. LOAD ORIGINAL RATINGS DATA #######################
# Interpret what the clusters mean.

# Bring back original ratings data
ratings <- read_delim(
  "C:/Users/terra/Documents/Schoolwork/SP26_Math252/252_project/data/ml-1m/ratings.dat",
  delim = "::",
  col_names = c("UserID", "MovieID", "Rating", "Timestamp"),
  show_col_types = FALSE
)

merged <- ratings %>%
  inner_join(cluster_df, by = "UserID")

movies <- read_delim(
  "C:/Users/terra/Documents/Schoolwork/SP26_Math252/252_project/data/ml-1m/ratings.dat",
  delim = "::",
  col_names = c("MovieID", "Title", "Genres"),
  show_col_types = FALSE
)
merged_movies <- merged %>%
  inner_join(movies, by = "MovieID")


####################### 12. MERGE RATINGS WITH CLUSTERS #######################

merged <- ratings %>%
  inner_join(cluster_df, by = "UserID")

head(merged)


####################### 13. INTERPRETATION: AVERAGE RATING BEHAVIOR #######################
# Compare average rating levels and total ratings by cluster

rating_summary <- merged %>%
  group_by(Cluster) %>%
  summarise(
    avg_rating = mean(Rating),
    sd_rating = sd(Rating),
    n_ratings = n(),
    .groups = "drop"
  )
#* this tells us: Do users in one cluster rate higher/lower?  Are they more consistent?

print(rating_summary)


####################### 14. INTERPRETATION: USER ACTIVITY #######################
# How many ratings per user in each cluster?

user_activity <- merged %>%
  group_by(UserID, Cluster) %>%
  summarise(
    num_ratings = n(),
    .groups = "drop"
  )

activity_summary <- user_activity %>%
  group_by(Cluster) %>%
  summarise(
    avg_ratings_per_user = mean(num_ratings),
    median_ratings_per_user = median(num_ratings),
    min_ratings_per_user = min(num_ratings),
    max_ratings_per_user = max(num_ratings),
    .groups = "drop"
  )

print(activity_summary)


####################### 15. INTERPRETATION: GENRE PREFERENCES #######################
# Merge in movie genres and summarize top genres by cluster

merged_movies <- merged %>%
  inner_join(movies, by = "MovieID")

# Split genre strings like "Action|Adventure|Sci-Fi" into separate rows
genre_data <- merged_movies %>%
  separate_rows(Genres, sep = "\\|")

genre_summary <- genre_data %>%
  group_by(Cluster, Genres) %>%
  summarise(
    count = n(),
    .groups = "drop"
  ) %>%
  group_by(Cluster) %>%
  mutate(
    prop = count / sum(count)
  ) %>%
  arrange(Cluster, desc(prop))

print(genre_summary)

# Top 5 genres per cluster
top_genres <- genre_summary %>%
  group_by(Cluster) %>%
  slice_head(n = 5)

print(top_genres)


####################### 16. PLOT GENRE PREFERENCES #######################

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


####################### 17. SAVE RESULTS #######################

write.csv(cluster_df, "output/kmeans_cluster_assignments.csv", row.names = FALSE)
write.csv(k_results, "output/kmeans_k_selection_metrics.csv", row.names = FALSE)
write.csv(rating_summary, "output/kmeans_rating_summary.csv", row.names = FALSE)
write.csv(activity_summary, "output/kmeans_activity_summary.csv", row.names = FALSE)
write.csv(genre_summary, "output/kmeans_genre_summary.csv", row.names = FALSE)


####################### NOTES FOR REPORTS #######################
#
# "The k-means results with k = 2 suggest that the embedding space contains one dominant user group and one smaller but distinct subgroup. The larger cluster represents the majority of users with more typical rating behavior, while the smaller cluster appears to capture users with distinct engagement patterns, rating tendencies, or genre preferences. Differences in activity level, average ratings, and genre proportions can be used to characterize the behavioral meaning of the two groups."