####################### 1. SETUP #######################
source("setup.R")


############# FOR TESTING ###########
embedding_path <- "output/embeddings/user_embeddings_64.csv"
embeddings <- read_csv(embedding_path, show_col_types = FALSE)
user_ids <- embeddings[[1]]          # Save user IDs, first column is UserID
embedding_matrix <- embeddings[, -1] # Remove UserID column. Remaining columns are embedding features

embedding_matrix <- scale(as.matrix(embedding_matrix)) # Convert to matrix and scale before clustering


####################### 2. TUNE CLUSTERS AND REDUCED DIMENSIONS #######################

## Final run
k_range <- 2:10
q_range <- 2:6



############### TESTING #############
k_range <- 2:6
q_range <- 2:4
#######################

x_mat <- as.matrix(embedding_matrix)

# *****Maybe cut later: use sampled silhouette during tuning for speed
use_sampled_silhouette <- TRUE
sil_sample_size <- 1500

if (use_sampled_silhouette) {
  set.seed(42)
  sil_idx <- sort(sample(seq_len(nrow(x_mat)), min(sil_sample_size, nrow(x_mat))))
  dmat_sub <- dist(x_mat[sil_idx, , drop = FALSE])
} else {
  dmat_full <- dist(x_mat)
}

############# PARALELLIZED ####################
library(future.apply)

plan(multisession)

grid <- expand.grid(k = k_range, q = q_range)

rkm_results_list <- future_lapply(seq_len(nrow(grid)), function(i) {
  k <- grid$k[i]
  q <- grid$q[i]
  
  model <- cluspca(
    x_mat,
    nclus = k,
    ndim = q,
    method = "RKM",
    rotation = "varimax"
  )
  
  labels <- model$cluster
  sil <- silhouette(labels[sil_idx], dmat_sub)
  avg_sil <- mean(sil[, 3])
  
  ch_val <- intCriteria(
    traj = x_mat,
    part = as.integer(labels),
    crit = "Calinski_Harabasz"
  )$calinski_harabasz
  
  list(
    result = data.frame(
      Method = "RKM",
      k = k,
      q = q,
      Silhouette = avg_sil,
      CH = ch_val
    ),
    model = model,
    labels = labels
  )
})















##################################33

rkm_results_list <- vector("list", length(k_range) * length(q_range))

best_model_obj <- NULL
best_score <- -Inf
counter <- 1

for (k in k_range) {
  for (q in q_range) {
    cat("Running RKM with k =", k, "and q =", q, "\n")
    
    model <- cluspca(
      x_mat,
      nclus = k,
      ndim = q,
      method = "RKM",
      rotation = "varimax"
    )
    
    labels <- model$cluster
    
    # Silhouette
    if (use_sampled_silhouette) {
      sil <- silhouette(labels[sil_idx], dmat_sub)
    } else {
      sil <- silhouette(labels, dmat_full)
    }
    avg_sil <- mean(sil[, 3])
    
    # CH
    ch_val <- intCriteria(
      traj = x_mat,
      part = as.integer(labels),
      crit = "Calinski_Harabasz"
    )$calinski_harabasz
    
    rkm_results_list[[counter]] <- data.frame(
      Method = "RKM",
      k = k,
      q = q,
      Silhouette = avg_sil,
      CH = ch_val
    )
    
    # Keep only the best model in memory
    # Primary criterion: silhouette
    # Secondary criterion: CH
    current_score <- avg_sil + 1e-6 * ch_val
    if (current_score > best_score) {
      best_score <- current_score
      best_model_obj <- list(
        model = model,
        labels = labels,
        avg_silhouette = avg_sil,
        CH = ch_val,
        k = k,
        q = q
      )
    }
    
    counter <- counter + 1
  }
}

####################### 3. RESULTS TABLE #######################

rkm_results <- dplyr::bind_rows(rkm_results_list) %>%
  arrange(desc(Silhouette), desc(CH))

print(rkm_results)

####################### 4. PLOT MODEL SELECTION #######################

ggplot(rkm_results, aes(x = q, y = Silhouette, color = factor(k), group = k)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  labs(
    title = "RKM Model Selection by Silhouette",
    x = "Reduced Dimension (q)",
    y = "Average Silhouette Width",
    color = "Clusters (k)"
  )

ggplot(rkm_results, aes(x = q, y = CH, color = factor(k), group = k)) +
  geom_line() +
  geom_point() +
  theme_minimal() +
  labs(
    title = "RKM Model Selection by Calinski-Harabasz",
    x = "Reduced Dimension (q)",
    y = "CH Index",
    color = "Clusters (k)"
  )

####################### 5. CHOOSE BEST MODEL #######################

best_k <- best_model_obj$k
best_q <- best_model_obj$q

cat("Best RKM model based on tuning:\n")
print(rkm_results[1, ])

best_model <- best_model_obj$model
best_labels <- best_model_obj$labels
final_avg_sil <- best_model_obj$avg_silhouette
final_ch <- best_model_obj$CH

####################### 6. FULL SILHOUETTE FOR FINAL MODEL #######################
# Even if tuning used sampled silhouette, compute full silhouette for final reporting

dmat_full <- dist(x_mat)
best_sil <- silhouette(best_labels, dmat_full)
final_avg_sil_full <- mean(best_sil[, 3])

plot(
  best_sil,
  main = paste("RKM Silhouette Plot for k =", best_k, ", q =", best_q),
  col = "blue",
  border = NA
)

cat("Cluster sizes for best RKM model (k =", best_k, ", q =", best_q, ")\n")
print(table(best_labels))

cat("Tuning silhouette:", final_avg_sil, "\n")
cat("Final full-data silhouette:", final_avg_sil_full, "\n")
cat("Final CH index:", final_ch, "\n")

####################### 7. FINAL CLUSTER ASSIGNMENTS #######################

cluster_df <- data.frame(
  UserID = user_ids,
  Cluster = best_labels
)

####################### 8. PCA VISUALIZATION #######################

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
    title = paste("Reduced K-Means (PCA Projection), k =", best_k, ", q =", best_q),
    x = paste0("PC1 (", round(100 * var_explained[1], 1), "%)"),
    y = paste0("PC2 (", round(100 * var_explained[2], 1), "%)")
  )

####################### 9. OPTIONAL: RKM OBJECT PLOT #######################

plot(best_model, cludesc = TRUE)

####################### 10. SAVE RESULTS #######################

dir.create("output", showWarnings = FALSE)

write.csv(cluster_df, "output/rkm_cluster_assignments.csv", row.names = FALSE)
write.csv(rkm_results, "output/rkm_model_selection_metrics.csv", row.names = FALSE)

saveRDS(best_model, "output/rkm_best_model.rds")
saveRDS(rkm_results, "output/rkm_results.rds")
saveRDS(cluster_df, "output/rkm_cluster_df.rds")
saveRDS(pca_df, "output/rkm_pca_df.rds")


####################### OPTIONAL: PCA SCREE PLOT FOR q #######################

outpca <- princomp(embedding_matrix)
plot(outpca, main = "PCA Scree Plot for Choosing q")

####################### OPTIONAL: PCA SCREE PLOT FOR q #######################

outpca <- princomp(embedding_matrix)
plot(outpca, main = "PCA Scree Plot for Choosing q")

plot(best_model, cludesc = TRUE)



