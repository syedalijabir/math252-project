
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
  "hopkins",
  "clustrd" # RKM requires clustrd
)

installed <- rownames(installed.packages())
missing_pkgs <- setdiff(required_packages, installed)
if (length(missing_pkgs) > 0) {
  install.packages(missing_pkgs)
}

library(readr); library(dplyr); library(tidyr); library(ggplot2)
library(cluster); library(clusterCrit); library(factoextra); 
library(FPDclustering); library(hopkins)
library(clustrd)

####################### MISC SETUP ####################### 
set.seed(42)
setwd("C:/Users/terra/Documents/Schoolwork/SP26_Math252/252_project/math252-project")

####################### 2. LOAD EMBEDDINGS FILES ####################### 

embedding_path <- "output/embeddings/user_embeddings_128.csv"
embeddings <- read_csv(embedding_path, show_col_types = FALSE)
user_ids <- embeddings[[1]]          # Save user IDs, first column is UserID
embedding_matrix <- embeddings[, -1] # Remove UserID column. Remaining columns are embedding features

embedding_matrix <- scale(as.matrix(embedding_matrix)) # Convert to matrix and scale before clustering

cat("Embedding matrix dimensions:", dim(embedding_matrix), "\n")

####################### 3. K CANDIDATES AND DISTANCE MATRIX ####################### 
k_vals <- 2:20
dmat <- dist(embedding_matrix) # Compute distance matrix once for silhouette


####################### 4. KMEANS: CHOOSE NUMBER OF CLUSTERS #######################
#* We evaluate k = 2 to 20 using:
  #* WSS (elbow method)
  #* average silhouette width
  #* Calinski-Harabasz (CH) index
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
