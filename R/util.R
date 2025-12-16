source("R/constants.R")
library(EBImage)

pad_to_square <- function(vec, target_size = target_image_size) {
  n <- length(vec)
  new_len <- target_size * target_size
  # Truncate if too long, pad if too short
  if (n > new_len) {
    padded <- vec[1:new_len]
  } else {
    padded <- c(vec, rep(0, new_len - n))
  }
  list(values = padded, side = target_size)
}

load_json_features <- function(filepath) {
  json_data <- jsonlite::fromJSON(filepath)
  
  # JSON keys are strings, values are arrays
  data_list <- lapply(json_data, function(x) as.numeric(x))
  
  return(data_list)
}

apply_transformation <- function(data, out_dir = NULL) {
    # Use constant if out_dir not specified
    if (is.null(out_dir)) {
      out_dir <- out_dir_default
    }
    # Create output directory
    dir.create(out_dir, showWarnings = FALSE)

    for (id in names(data)) {

        vec <- data[[id]]

        pad <- pad_to_square(vec)
        padded_vec <- pad$values
        side <- pad$side

        # reshape
        img_matrix <- matrix(padded_vec, nrow = side, ncol = side, byrow = TRUE)

        # normalize using max feature = 4000
        img_norm <- img_matrix / 4000

        # create grayscale image
        img <- Image(img_norm)

        # save
        filepath <- sprintf("%s/entry_%s.png", out_dir, id)
        writeImage(img, filepath)

        cat("Saved:", filepath, "(", side, "x", side, ")\n")
    }
    
    return(out_dir)
}

load_target_data <- function(filepath) {
  # Load target labels from CSV
  target_df <- read.csv(filepath)
  
  # Convert to named numeric vector (id -> ml_target)
  # Column 1 is id, column 3 is ml_target
  target_vector <- setNames(as.numeric(target_df[[3]]), target_df[[1]])
  
  return(target_vector)
}

# Split dataset into train and test
split_dataset <- function(images_list, image_ids, targets,
                          tabular_df, pca_df,
                          split_ratio = 0.7, seed = 123) {
  set.seed(seed)
  
  # Only consider samples that have targets
  valid_indices <- which(image_ids %in% names(targets))
  valid_ids <- image_ids[valid_indices]
  
  n_samples <- length(valid_indices)
  n_train <- floor(n_samples * split_ratio)
  
  train_indices <- sample(valid_indices, n_train)
  test_indices <- setdiff(valid_indices, train_indices)
  
  train_data <- list(
    images = images_list[train_indices],
    ids = image_ids[train_indices],
    targets = targets[image_ids[train_indices]],
    tabular_df = tabular_df[train_indices, ],
    pca_df = pca_df[train_indices, ]
  )
  
  test_data <- list(
    images = images_list[test_indices],
    ids = image_ids[test_indices],
    targets = targets[image_ids[test_indices]],
    tabular_df = tabular_df[test_indices, ],
    pca_df = pca_df[test_indices, ]
  )
  
  return(list(train = train_data, test = test_data))
}

prepare_tabular <- function(git_features_data, target_labels) {
  all_features <- sort(unique(unlist(git_features_data)))
  
  feature_matrix <- matrix(0, nrow = length(git_features_data), ncol = length(all_features))
  colnames(feature_matrix) <- as.character(all_features)
  rownames(feature_matrix) <- names(git_features_data)
  
  for (id in names(git_features_data)) {
    features <- git_features_data[[id]]
    feat_cols <- as.character(features)
    feat_cols <- feat_cols[feat_cols %in% colnames(feature_matrix)]
    
    if (length(feat_cols) > 0) {
      feature_matrix[id, feat_cols] <- 1
    }
  }
  
  feature_df <- as.data.frame(feature_matrix)
  
  ids <- names(git_features_data)
  feature_df$id <- ids
  feature_df$target <- target_labels[ids]
  
  return(feature_df)
}

apply_pca <- function(tabular_df, n_components = 32) {
  feature_cols <- setdiff(colnames(tabular_df), c("id", "target"))
  feature_matrix <- as.matrix(tabular_df[, feature_cols])
  
  pca_result <- prcomp(feature_matrix, center = TRUE, scale. = TRUE)
  
  pca_scores <- as.data.frame(pca_result$x[, 1:n_components])
  pca_scores$id <- tabular_df$id
  pca_scores$target <- tabular_df$target
  
  return(pca_scores)
}