# Load and prepare target data
load_target_data <- function(filepath) {
  target_df <- read.csv(filepath)
  # Return as named vector: names are IDs, values are ml_target
  targets <- setNames(target_df$ml_target, as.character(target_df$id))
  return(targets)
}

# Extract image features (simple statistics)
extract_image_features <- function(image_path) {
  img <- EBImage::readImage(image_path)
  features <- c(
    mean = mean(img),
    sd = sd(as.numeric(img)),
    min = min(img),
    max = max(img),
    median = median(as.numeric(img))
  )
  return(features)
}

# Convert images to feature vectors
images_to_features <- function(image_dir, target_image_size = 7) {
  # List all PNG files
  png_files <- list.files(image_dir, pattern = "\\.png$", full.names = TRUE)
  
  if (length(png_files) == 0) {
    stop("No PNG files found in ", image_dir)
  }
  
  # Extract features from each image
  n_images <- length(png_files)
  features_list <- list()
  image_ids <- character(n_images)
  
  for (i in seq_along(png_files)) {
    filename <- basename(png_files[i])
    image_ids[i] <- gsub("entry_|.png", "", filename)
    features_list[[i]] <- extract_image_features(png_files[i])
  }
  
  # Combine into matrix
  features_matrix <- do.call(rbind, features_list)
  rownames(features_matrix) <- image_ids
  
  return(list(features = features_matrix, ids = image_ids))
}

# Simple logistic regression classifier
train_and_predict <- function(features_matrix, image_ids, targets, test_split = 0.2) {
  # Match images with targets
  y_labels <- targets[image_ids]
  
  # Handle missing targets
  valid_indices <- !is.na(y_labels)
  X_valid <- features_matrix[valid_indices, ]
  y_valid <- as.numeric(y_labels[valid_indices])
  valid_ids <- image_ids[valid_indices]
  
  cat("Training data:", sum(valid_indices), "images\n")
  cat("Features per image:", ncol(X_valid), "\n")
  
  # Split into train/test
  n_train <- floor(nrow(X_valid) * (1 - test_split))
  train_idx <- 1:n_train
  test_idx <- (n_train + 1):nrow(X_valid)
  
  # Fit logistic regression model
  train_data <- as.data.frame(cbind(X_valid[train_idx, ], y = y_valid[train_idx]))
  model <- glm(y ~ ., data = train_data, family = "binomial")
  
  # Predictions on all data
  all_data <- as.data.frame(X_valid)
  predictions_prob <- predict(model, all_data, type = "response")
  predictions_binary <- ifelse(predictions_prob > 0.5, 1, 0)
  
  # Calculate metrics
  accuracy <- mean(predictions_binary == y_valid)
  
  # Separate train and test accuracy
  if (length(test_idx) > 0) {
    train_accuracy <- mean(predictions_binary[train_idx] == y_valid[train_idx])
    test_accuracy <- mean(predictions_binary[test_idx] == y_valid[test_idx])
    cat("Train Accuracy:", train_accuracy, "\n")
    cat("Test Accuracy:", test_accuracy, "\n")
  } else {
    cat("Overall Accuracy:", accuracy, "\n")
  }
  
  # Return results with IDs
  results <- data.frame(
    id = valid_ids,
    actual = y_valid,
    predicted = predictions_binary,
    probability = predictions_prob
  )
  
  return(list(
    model = model,
    results = results,
    accuracy = accuracy
  ))
}
