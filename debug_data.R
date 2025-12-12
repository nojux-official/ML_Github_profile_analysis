#!/usr/bin/env Rscript
# Debug script to inspect data loading and preprocessing

source("R/constants.R")
source("R/util.R")
library(EBImage)
library(jsonlite)

cat("=== DEBUGGING DATA LOADING ===\n\n")

# 1. Load JSON features
cat("1. Loading JSON features...\n")
json_file <- "dataset/musae_git_features.json"
json_data <- fromJSON(json_file)
cat("  - Number of samples:", length(json_data), "\n")
cat("  - Sample IDs (first 5):", paste(names(json_data)[1:5], collapse=", "), "\n")

# Check feature vectors
sample_features <- json_data[[1]]
cat("  - Feature vector length (first sample):", length(sample_features), "\n")
cat("  - Feature vector range:", min(sample_features), "to", max(sample_features), "\n")
cat("  - Feature vector stats: mean=", mean(sample_features), ", sd=", sd(sample_features), "\n\n")

# 2. Check image transformation
cat("2. Checking image transformation...\n")
data_list <- lapply(json_data, function(x) as.numeric(x))

# Apply transformation
dir.create(out_dir, showWarnings = FALSE)
cat("  - Output directory:", out_dir, "\n")

# Check a few samples before and after transformation
for (i in 1:3) {
  id <- names(data_list)[i]
  vec <- data_list[[id]]
  
  # Pad to square
  pad <- pad_to_square(vec)
  padded_vec <- pad$values
  side <- pad$side
  
  cat("  - Sample", id, ":\n")
  cat("    Original vector length:", length(vec), "\n")
  cat("    Original range:", min(vec), "to", max(vec), "\n")
  cat("    Padded vector length:", length(padded_vec), "\n")
  cat("    Padded range:", min(padded_vec), "to", max(padded_vec), "\n")
  
  # Reshape to matrix
  img_matrix <- matrix(padded_vec, nrow = side, ncol = side, byrow = TRUE)
  cat("    Matrix shape:", nrow(img_matrix), "x", ncol(img_matrix), "\n")
  
  # Normalize
  img_norm <- img_matrix / 4000
  cat("    After normalization range:", min(img_norm), "to", max(img_norm), "\n")
  cat("    After normalization mean:", mean(img_norm), ", sd:", sd(img_norm), "\n\n")
}

# 3. Load target data
cat("3. Loading target data...\n")
target_file <- "dataset/musae_git_target.csv"
target_df <- read.csv(target_file)
cat("  - Target CSV dimensions:", nrow(target_df), "x", ncol(target_df), "\n")
cat("  - Column names:", paste(colnames(target_df), collapse=", "), "\n")
cat("  - First 5 rows:\n")
print(head(target_df, 5))

# Convert to named vector
target_vector <- setNames(target_df[[2]], target_df[[1]])
cat("  - Target vector length:", length(target_vector), "\n")
cat("  - Target value range:", min(target_vector, na.rm=TRUE), "to", max(target_vector, na.rm=TRUE), "\n")
cat("  - Target unique values:", paste(sort(unique(target_vector)), collapse=", "), "\n")
cat("  - NA count:", sum(is.na(target_vector)), "\n\n")

# 4. Check image files after transformation
cat("4. Checking saved PNG files...\n")
png_files <- list.files(out_dir, pattern = "\\.png$", full.names = TRUE)
cat("  - Number of PNG files:", length(png_files), "\n")

if (length(png_files) > 0) {
  # Load and check a few images
  for (i in 1:min(3, length(png_files))) {
    img <- EBImage::readImage(png_files[i])
    filename <- basename(png_files[i])
    img_id <- gsub("entry_|.png", "", filename)
    
    cat("  - File", filename, ":\n")
    cat("    Dimensions:", paste(dim(img), collapse=" x "), "\n")
    cat("    Value range:", min(img), "to", max(img), "\n")
    cat("    Mean:", mean(img), ", SD:", sd(img), "\n")
    
    # Check if grayscale
    if (length(dim(img)) > 2) {
      cat("    WARNING: Image has", dim(img)[3], "channels\n")
    } else {
      cat("    Grayscale image: OK\n")
    }
    
    # Convert to matrix
    img_matrix <- as.matrix(img)
    cat("    Matrix range after as.matrix:", min(img_matrix), "to", max(img_matrix), "\n")
    cat("    Matrix clamped range:", min(pmin(pmax(img_matrix, 0), 1)), "to", max(pmin(pmax(img_matrix, 0), 1)), "\n\n")
  }
}

# 5. Match images to targets
cat("5. Matching images to targets...\n")
image_ids <- gsub("entry_|.png", "", basename(png_files))
cat("  - Image IDs (first 5):", paste(image_ids[1:5], collapse=", "), "\n")

# Check how many images have target labels
matches <- image_ids %in% names(target_vector)
cat("  - Images with matching targets:", sum(matches), "out of", length(image_ids), "\n")

# Check label distribution for images with targets
matched_labels <- target_vector[image_ids[matches]]
cat("  - Matched target value range:", min(matched_labels, na.rm=TRUE), "to", max(matched_labels, na.rm=TRUE), "\n")
cat("  - Matched target distribution:\n")
print(table(matched_labels))

cat("\n=== DEBUG COMPLETE ===\n")
