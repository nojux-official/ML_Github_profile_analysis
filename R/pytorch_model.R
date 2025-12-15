source("R/constants.R")
library(torch)
library(luz)

# Simple CNN for 6x6 grayscale image classification
create_simple_cnn <- function() {
  nn_module(
    "SimpleCNN",
    initialize = function() {
      self$conv1 <- nn_conv2d(1, 16, kernel_size = 3, padding = 1)
      self$bn1 <- nn_batch_norm2d(16)
      self$pool <- nn_max_pool2d(2, 2)
      self$conv2 <- nn_conv2d(16, 32, kernel_size = 3, padding = 1)
      self$bn2 <- nn_batch_norm2d(32)
      
      # After 2 pooling layers: 6 -> 3 -> 1
      flat_size <- 32 * 1 * 1
      
      self$fc1 <- nn_linear(flat_size, 64)
      self$dropout <- nn_dropout(p = 0.3)
      self$fc2 <- nn_linear(64, 1)
      
      # Initialize weights with smaller values
      nn_init_kaiming_normal_(self$conv1$weight)
      nn_init_kaiming_normal_(self$conv2$weight)
      nn_init_xavier_uniform_(self$fc1$weight)
      nn_init_xavier_uniform_(self$fc2$weight)
      
      nn_init_constant_(self$fc2$bias, 0)
    },
    forward = function(x) {
      x %>%
        self$conv1() %>%
        self$bn1() %>%
        nnf_relu() %>%
        self$pool() %>%
        self$conv2() %>%
        self$bn2() %>%
        nnf_relu() %>%
        self$pool() %>%
        torch_flatten(start_dim = 2) %>%
        self$fc1() %>%
        nnf_relu() %>%
        self$dropout() %>%
        self$fc2()
    }
  )
}

# Convert images to torch tensors
prepare_pytorch_data <- function(image_dir, target_image_size = 6) {
  png_files <- list.files(image_dir, pattern = "\\.png$", full.names = TRUE)
  
  if (length(png_files) == 0) {
    stop("No PNG files found in ", image_dir)
  }
  
  # Initialize lists to store data
  n_images <- length(png_files)
  images_list <- list()
  image_ids <- character(n_images)
  
  for (i in seq_along(png_files)) {
    img <- EBImage::readImage(png_files[i])
    
    # Extract grayscale channel if multichannel
    if (length(dim(img)) > 2) {
      img <- img[, , 1]
    }
    
    # Resize to target size
    img <- EBImage::resize(img, w = target_image_size, h = target_image_size)
    
    # Extract filename (ID)
    filename <- basename(png_files[i])
    image_ids[i] <- gsub("entry_|.png", "", filename)
    
    # Convert to matrix and ensure values in [0, 1]
    img_matrix <- as.matrix(img)
    img_matrix <- pmin(pmax(img_matrix, 0), 1)
    images_list[[i]] <- img_matrix
  }
  
  return(list(images = images_list, ids = image_ids))
}

# Build and train CNN using native R torch
train_pytorch_cnn <- function(images_list, image_ids, targets, 
                             epochs = 20, batch_size = 32, learning_rate = 0.0001, threshold = cnn_train_threshold, model_name = "cnn_model") {
  
  # Prepare data
  y_labels <- targets[image_ids]
  valid_indices <- !is.na(y_labels)
  
  X_valid <- images_list[valid_indices]
  y_valid <- as.numeric(y_labels[valid_indices])
  valid_ids <- image_ids[valid_indices]
  
  # Ensure y_valid is binary (0 or 1)
  y_valid <- as.numeric(y_valid > 0)
  
  cat("Training samples:", length(y_valid), "\n")
  cat("Class distribution:", table(y_valid), "\n")
  
  # Convert to tensor format (N, 1, H, W)
  n_samples <- length(X_valid)
  h <- nrow(X_valid[[1]])
  w <- ncol(X_valid[[1]])
  
  X_array <- array(0, dim = c(n_samples, 1, h, w))
  for (i in seq_along(X_valid)) {
    X_array[i, 1, , ] <- X_valid[[i]]
  }
  
  # Standardize input data
  mean_val <- mean(X_array)
  std_val <- sd(as.numeric(X_array))
  if (std_val < 1e-6) std_val <- 1  # Avoid division by zero
  X_array <- (X_array - mean_val) / std_val
  
  X_tensor <- torch_tensor(X_array, dtype = torch_float32())
  y_tensor <- torch_tensor(y_valid, dtype = torch_float32())$unsqueeze(2)
  
  # Create dataset
  dataset <- torch::dataset(
    name = "image_dataset",
    initialize = function(x, y) {
      self$x <- x
      self$y <- y
    },
    .getitem = function(index) {
      list(x = self$x[index, , , ], y = self$y[index, ])
    },
    .length = function() {
      self$x$shape[1]
    }
  )
  
  train_ds <- dataset(X_tensor, y_tensor)
  train_dl <- torch::dataloader(train_ds, batch_size = batch_size, shuffle = TRUE)
  
  # architecture
  model <- create_simple_cnn()
  
  # using luz to train the model
  fitted_model <- model %>%
    setup(
      loss = nn_mse_loss(),
      optimizer = optim_adam
    ) %>%
    fit(
      data = train_dl,
      epochs = epochs,
      verbose = TRUE
    )
  
  trained_model <- fitted_model$model
  
  # Make predictions on full dataset
  trained_model$eval()
  with_no_grad({
    predictions <- trained_model(X_tensor)
  })
  
  predictions_numeric <- as.numeric(predictions)
  # Apply sigmoid for probability and threshold
  predictions_prob <- 1 / (1 + exp(-predictions_numeric))
  pred_binary <- ifelse(predictions_prob > threshold, 1, 0)
  
  # Calculate accuracy
  accuracy <- mean(pred_binary == y_valid)
  cat("\nAccuracy:", accuracy, "\n")
  
  # Calculate detailed metrics
  tp <- sum((pred_binary == 1) & (y_valid == 1))
  tn <- sum((pred_binary == 0) & (y_valid == 0))
  fp <- sum((pred_binary == 1) & (y_valid == 0))
  fn <- sum((pred_binary == 0) & (y_valid == 1))
  
  sensitivity <- if (tp + fn > 0) tp / (tp + fn) else NA
  specificity <- if (tn + fp > 0) tn / (tn + fp) else NA
  precision <- if (tp + fp > 0) tp / (tp + fp) else NA
  f1 <- if (!is.na(precision) && !is.na(sensitivity) && (precision + sensitivity) > 0) {
    2 * (precision * sensitivity) / (precision + sensitivity)
  } else NA
  
  metrics <- list(
    accuracy = accuracy,
    sensitivity = sensitivity,
    specificity = specificity,
    precision = precision,
    f1 = f1,
    tp = tp, tn = tn, fp = fp, fn = fn
  )
  
  results <- data.frame(
    id = valid_ids,
    actual = y_valid,
    predicted = pred_binary,
    probability = predictions_prob
  )
  
  save_model_to_disk(trained_model, model_save_dir, model_name)
  
  return(list(
    model = trained_model,
    results = results,
    metrics = metrics,
    accuracy = accuracy,
    fitted = fitted_model
  ))
}

save_model_to_disk <- function(model, save_dir = "static", model_name = "cnn_model") {
  if (!dir.exists(save_dir)) {
    dir.create(save_dir, showWarnings = FALSE, recursive = TRUE)
  }
  
  model_path <- file.path(save_dir, paste0(model_name, ".pt"))
  torch_save(model, model_path)
  
  cat("Model saved to:", model_path, "\n")
  
  return(model_path)
}

load_model_from_disk <- function(model_path = "static/cnn_model.pt") {
  if (!file.exists(model_path)) {
    stop("Model file not found at: ", model_path)
  }
  
  model <- torch_load(model_path)
  cat("Model loaded from:", model_path, "\n")
  
  return(model)
}

# Execute prediction on a single image
predict_image <- function(model, image_path, target_image_size = 6, threshold = cnn_eval_threshold) {
  img <- EBImage::readImage(image_path)
  
  # Extract grayscale channel if multichannel
  if (length(dim(img)) > 2) {
    img <- img[, , 1]
  }
  
  img <- EBImage::resize(img, w = target_image_size, h = target_image_size)
  
  # values in [0, 1]
  img_matrix <- as.matrix(img)
  img_matrix <- pmin(pmax(img_matrix, 0), 1)
  
  # Standardize (using global mean/std - in production, should use training set stats)
  mean_val <- mean(img_matrix)
  std_val <- sd(as.numeric(img_matrix))
  if (std_val < 1e-6) std_val <- 1
  img_matrix <- (img_matrix - mean_val) / std_val
  
  # Convert to tensor (1, 1, H, W)
  X_tensor <- torch_tensor(img_matrix, dtype = torch_float32())$unsqueeze(1)$unsqueeze(1)
  
  model$eval()
  with_no_grad({
    prediction <- model(X_tensor)
  })
  
  prediction_numeric <- as.numeric(prediction)
  # Threshold
  prediction_prob <- 1 / (1 + exp(-prediction_numeric))
  pred_binary <- ifelse(prediction_prob > threshold, 1, 0)
  
  return(list(
    probability = prediction_prob,
    prediction = pred_binary,
    raw_output = prediction_numeric
  ))
}

# Evaluate model on a dataset
evaluate_model <- function(model, images_list, image_ids, targets, threshold = cnn_eval_threshold) {
  
  # Prepare data
  y_labels <- targets[image_ids]
  valid_indices <- !is.na(y_labels)
  
  if (sum(valid_indices) == 0) {
    return(NULL)
  }
  
  X_valid <- images_list[valid_indices]
  y_valid <- as.numeric(y_labels[valid_indices])
  valid_ids <- image_ids[valid_indices]
  
  # Ensure y_valid is binary (0 or 1)
  y_valid <- as.numeric(y_valid > 0)
  
  # Convert to tensor format (N, 1, H, W)
  n_samples <- length(X_valid)
  h <- nrow(X_valid[[1]])
  w <- ncol(X_valid[[1]])
  
  X_array <- array(0, dim = c(n_samples, 1, h, w))
  for (i in seq_along(X_valid)) {
    X_array[i, 1, , ] <- X_valid[[i]]
  }
  
  # Standardize input data
  mean_val <- mean(X_array)
  std_val <- sd(as.numeric(X_array))
  if (std_val < 1e-6) std_val <- 1
  X_array <- (X_array - mean_val) / std_val
  
  X_tensor <- torch_tensor(X_array, dtype = torch_float32())
  
  # Make predictions
  model$eval()
  with_no_grad({
    predictions <- model(X_tensor)
  })
  
  predictions_numeric <- as.numeric(predictions)
  predictions_prob <- 1 / (1 + exp(-predictions_numeric))
  pred_binary <- ifelse(predictions_prob > threshold, 1, 0)
  
  # Calculate metrics
  accuracy <- mean(pred_binary == y_valid)
  
  tp <- sum((pred_binary == 1) & (y_valid == 1))
  tn <- sum((pred_binary == 0) & (y_valid == 0))
  fp <- sum((pred_binary == 1) & (y_valid == 0))
  fn <- sum((pred_binary == 0) & (y_valid == 1))
  
  sensitivity <- if (tp + fn > 0) tp / (tp + fn) else NA
  specificity <- if (tn + fp > 0) tn / (tn + fp) else NA
  precision <- if (tp + fp > 0) tp / (tp + fp) else NA
  f1 <- if (!is.na(precision) && !is.na(sensitivity) && (precision + sensitivity) > 0) {
    2 * (precision * sensitivity) / (precision + sensitivity)
  } else NA
  
  metrics <- list(
    accuracy = accuracy,
    sensitivity = sensitivity,
    specificity = specificity,
    precision = precision,
    f1 = f1,
    tp = tp, tn = tn, fp = fp, fn = fn
  )
  
  results <- data.frame(
    id = valid_ids,
    actual = y_valid,
    predicted = pred_binary,
    probability = predictions_prob
  )
  
  return(list(
    results = results,
    metrics = metrics,
    accuracy = accuracy
  ))
}

# Wrapper to run experiment for a given number of epochs
run_cnn_experiment <- function(data_split, epochs, learning_rate = 0.001, batch_size = 16) {
  model_name <- paste0("cnn_model_epochs_", epochs)
  
  # Train
  train_results <- train_pytorch_cnn(
    images_list = data_split$train$images,
    image_ids = data_split$train$ids,
    targets = data_split$train$targets,
    epochs = epochs,
    batch_size = batch_size,
    learning_rate = learning_rate,
    threshold = cnn_train_threshold,
    model_name = model_name
  )
  
  # Evaluate
  test_results <- evaluate_model(
    model = train_results$model,
    images_list = data_split$test$images,
    image_ids = data_split$test$ids,
    targets = data_split$test$targets
  )
  
  list(
    epochs = epochs,
    train = train_results,
    test = test_results,
    model_name = model_name
  )
}

# Run CNN experiment with given epoch count
run_cnn_experiment <- function(data_split, epochs, batch_size = 16, learning_rate = 0.001, threshold = cnn_train_threshold) {
  model_name <- paste0("cnn_model_", epochs, "ep")
  
  model_result <- train_pytorch_cnn(
    data_split$train$images,
    data_split$train$ids,
    data_split$train$targets,
    epochs = epochs,
    batch_size = batch_size,
    learning_rate = learning_rate,
    threshold = threshold,
    model_name = model_name
  )
  
  # Evaluate on test set
  test_result <- evaluate_model(
    model_result$model,
    data_split$test$images,
    data_split$test$ids,
    data_split$test$targets
  )
  
  # Create model_performance object
  model_perf <- list(
    epochs = epochs,
    train_results = model_result$results,
    test_results = test_result$results,
    train_metrics = model_result$metrics,
    test_metrics = test_result$metrics,
    model_name = model_name,
    train_accuracy = model_result$accuracy,
    test_accuracy = test_result$accuracy
  )
  class(model_perf) <- "model_performance"
  
  return(model_perf)
}
