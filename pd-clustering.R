
# Requirements

required_packages <- c(
  "readr", "dplyr", "tidyr", "ggplot2",
  "cluster", "factoextra",
  "FPDclustering"
)

installed <- rownames(installed.packages())
missing_pkgs <- setdiff(required_packages, installed)
if (length(missing_pkgs) > 0) {
  install.packages(missing_pkgs)
}

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(cluster)
library(factoextra)
library(clustertend)
library(FPDclustering)

# 1. Load embeddings files

# Change to whichever embedding size you want
embedding_path <- "output/embeddings/user_embeddings_128.csv"
embeddings <- readr::read_csv(embedding_path)
user_ids <- embeddings[[1]]          # Save user IDs
embedding_matrix <- embeddings[, -1] # Remove UserID column

embedding_matrix <- as.matrix(embedding_matrix)



# 2. Choose Number of Clusters

results <- list()
avg_silhouette <- numeric()

for (k in 2:20) {
  cat("Running PDC for k =", k, "\n")
  
  # Run PD Clustering
  pdc_model <- PDC(embedding_matrix, k = k)
  
  # Extract labels
  labels <- pdc_model$label
  
  # Compute distance matrix
  d <- dist(embedding_matrix)
  
  # Compute silhouette
  sil <- silhouette(labels, d)
  
  #Silh(pdc_model$probability)
  
  # Store average silhouette width
  avg_sil <- mean(sil[, 3])
  avg_silhouette[k] <- avg_sil
  
  # Save everything
  results[[k]] <- list(
    model = pdc_model,
    silhouette = sil,
    avg_silhouette = avg_sil
  )
  
  cat("Avg silhouette for k =", k, ":", avg_sil, "\n\n")
}

# 2a. Plot Silhouette
ks <- 2:20
plot(ks, avg_silhouette[ks],
     type = "b",
     pch = 19,
     col = "blue",
     xlab = "Number of Clusters (K)",
     ylab = "Average Silhouette Width",
     main = "PD-Clustering Model Selection")


# 2b. Plot Silhouettes
par(mfrow = c(4, 5), mar = c(2, 2, 2, 1))

for (k in 2:20) {
  sil <- results[[k]]$silhouette
  
  plot(sil,
       main = paste("K =", k),
       col = "blue",
       border = NA)
}

# Reset layout
par(mfrow = c(1,1))


# 3. Choose best PD clustering

set.seed(123)

best_k <- 4 #which.max(avg_silhouette)

plot(results[[best_k]]$silhouette,
     main = paste("Silhouette Plot for K =", best_k),
     col = "blue",
     border = NA)


# 4. Visualize with PCA
pca <- prcomp(embedding_matrix)

pca_df <- data.frame(
  PC1 = pca$x[,1],
  PC2 = pca$x[,2],
  Cluster = as.factor(results[[best_k]]$model$label)
)

ggplot(pca_df, aes(x=PC1, y=PC2, color=Cluster)) +
  geom_point(alpha=0.6) +
  theme_minimal() +
  ggtitle("K-Means Clusters (PCA Projection)")

ggplot(pca_df[pca_df$Cluster == 1, ], 
       aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(alpha = 0.6) +
  theme_minimal() +
  ggtitle("Cluster 1 (PCA Projection)")

ggplot(pca_df[pca_df$Cluster == 2, ], 
       aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(alpha = 0.6) +
  theme_minimal() +
  ggtitle("Cluster 1 (PCA Projection)")

ggplot(pca_df[pca_df$Cluster == 3, ], 
       aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(alpha = 0.6) +
  theme_minimal() +
  ggtitle("Cluster 1 (PCA Projection)")

