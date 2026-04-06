
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

