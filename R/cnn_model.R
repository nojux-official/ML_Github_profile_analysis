source("R/constants.R")
library(torch)
library(luz)


   ##   #####   ####  #    # # ##### ######  ####  ##### #    # #####  ######
  #  #  #    # #    # #    # #   #   #      #    #   #   #    # #    # #
 #    # #    # #      ###### #   #   #####  #        #   #    # #    # #####
 ###### #####  #      #    # #   #   #      #        #   #    # #####  #
 #    # #   #  #    # #    # #   #   #      #    #   #   #    # #   #  #
 #    # #    #  ####  #    # #   #   ######  ####    #    ####  #    # ######

# Simple CNN for 6x6 grayscale image classification
create_simple_cnn <- function(dropout_rate = cnn_dropout_rate) {
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
      self$dropout <- nn_dropout(p = dropout_rate)
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



 ##### #####    ##   # #    #
   #   #    #  #  #  # ##   #
   #   #    # #    # # # #  #
   #   #####  ###### # #  # #
   #   #   #  #    # # #   ##
   #   #    # #    # # #    #

# Build and train CNN using native R torch
train_pytorch_cnn <- function(images_list, image_ids, targets, 
                             epochs = 20, batch_size = 32, learning_rate = 0.0001, 
                             dropout_rate = 0.3, use_class_weights = TRUE,
                             threshold = cnn_train_threshold, model_name = "cnn_model") {
  
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
  
  # Calculate class weights if requested
  if (use_class_weights) {
    n_pos <- sum(y_valid == 1)
    n_neg <- sum(y_valid == 0)
    # Weight for positive class to balance the loss
    pos_weight_val <- if(n_pos > 0) n_neg / n_pos else 1
    cat("Using class weighting. Positive class weight:", pos_weight_val, "\n")
    loss_fn <- nn_bce_with_logits_loss(pos_weight = torch_tensor(pos_weight_val))
  } else {
    loss_fn <- nn_bce_with_logits_loss()
  }
  
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
  model <- create_simple_cnn(dropout_rate = dropout_rate)
  
  # using luz to train the model
  fitted_model <- model %>%
    setup(
      loss = loss_fn,
      optimizer = optim_adam
    ) %>%
    set_opt_hparams(lr = learning_rate) %>%
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
  
  sensitivity <- if (tp + fn > 0) tp / (tp + fn) else 0
  specificity <- if (tn + fp > 0) tn / (tn + fp) else 0
  precision <- if (tp + fp > 0) tp / (tp + fp) else 0
  f1 <- if ((precision + sensitivity) > 0) {
    2 * (precision * sensitivity) / (precision + sensitivity)
  } else 0
  
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

 ####### ####### #       ######
 #       #     # #       #     #
 #       #     # #       #     #
 #####   #     # #       #     #
 #       #     # #       #     #
 #       #     # #       #     #
 #       ####### ####### ######

# Run CNN experiment with 5-fold cross-validation
run_cnn_experiment <- function(data_split, epochs, batch_size = 16, learning_rate = 0.001, threshold = cnn_train_threshold, n_folds = 1) {
  model_name <- paste0("cnn_model_", epochs, "ep")
  
  # Combine train and test data for k-fold CV
  all_images <- c(data_split$train$images)
  all_ids <- c(data_split$train$ids)
  all_targets <- c(data_split$train$targets)
  
  # Create fold indices
  n_samples <- length(all_ids)
  fold_indices <- cut(seq(1, n_samples), breaks = n_folds, labels = FALSE)
  fold_indices <- fold_indices[sample(seq(1, n_samples))]  # Shuffle folds
  
  # Storage for cross-validation results
  cv_train_results <- list()
  cv_val_results <- list()
  cv_train_metrics <- list()
  cv_val_metrics <- list()
  cv_models <- list()
  
  cat("Starting 5-fold cross-validation with", epochs, "epochs\n")
  cat("Total samples:", n_samples, "\n\n")
  
  # Perform k-fold CV
  for (fold in 1:n_folds) {
    cat("=== Fold", fold, "of", n_folds, "===\n")
    
    # Create train/validation split for this fold
    val_indices <- which(fold_indices == fold)
    train_indices <- which(fold_indices != fold)
    
    fold_train_images <- all_images[train_indices]
    fold_train_ids <- all_ids[train_indices]
    fold_train_targets <- all_targets[train_indices]
    
    fold_val_images <- all_images[val_indices]
    fold_val_ids <- all_ids[val_indices]
    fold_val_targets <- all_targets[val_indices]
    
    cat("Train samples:", length(fold_train_ids), "| Validation samples:", length(fold_val_ids), "\n")
    
    # Train model on this fold
    fold_model_name <- paste0(model_name, "_fold", fold)
    fold_train_result <- train_pytorch_cnn(
      images_list = fold_train_images,
      image_ids = fold_train_ids,
      targets = fold_train_targets,
      epochs = epochs,
      batch_size = batch_size,
      learning_rate = learning_rate,
      threshold = threshold,
      model_name = fold_model_name
    )
    
    # Evaluate on validation set
    fold_val_result <- evaluate_model(
      model = fold_train_result$model,
      images_list = fold_val_images,
      image_ids = fold_val_ids,
      targets = fold_val_targets
    )
    
    # Store results
    cv_train_results[[fold]] <- fold_train_result$results
    cv_val_results[[fold]] <- fold_val_result$results
    cv_train_metrics[[fold]] <- fold_train_result$metrics
    cv_val_metrics[[fold]] <- fold_val_result$metrics
    cv_models[[fold]] <- fold_train_result$model
    
    cat("Train Accuracy:", round(fold_train_result$accuracy, 4), 
        "| Validation Accuracy:", round(fold_val_result$accuracy, 4), "\n")
    cat("Validation F1:", round(fold_val_result$metrics$f1, 4), "\n\n")
  }
  
  # Combine results across all folds
  combined_val_results <- do.call(rbind, cv_val_results)
  rownames(combined_val_results) <- NULL
  
  # Calculate aggregate metrics
  all_train_accuracies <- sapply(cv_train_metrics, function(x) x$accuracy)
  all_val_accuracies <- sapply(cv_val_metrics, function(x) x$accuracy)
  all_val_f1s <- sapply(cv_val_metrics, function(x) x$f1)
  
  mean_train_acc <- mean(all_train_accuracies)
  mean_val_acc <- mean(all_val_accuracies)
  sd_val_acc <- sd(all_val_accuracies)
  mean_val_f1 <- mean(all_val_f1s)
  
  cat("===== Cross-Validation Summary =====\n")
  cat("Mean Train Accuracy:", round(mean_train_acc, 4), "\n")
  cat("Mean Validation Accuracy:", round(mean_val_acc, 4), "±", round(sd_val_acc, 4), "\n")
  cat("Mean Validation F1:", round(mean_val_f1, 4), "\n")
  cat("Individual fold accuracies:", paste(round(all_val_accuracies, 4), collapse = ", "), "\n\n")
  
  # Create model_performance object with CV results
  model_perf <- list(
    epochs = epochs,
    cv_train_results = cv_train_results,
    cv_val_results = cv_val_results,
    combined_val_results = combined_val_results,
    cv_train_metrics = cv_train_metrics,
    cv_val_metrics = cv_val_metrics,
    cv_models = cv_models,
    model_name = model_name,
    n_folds = n_folds,
    fold_accuracies = all_val_accuracies,
    mean_train_accuracy = mean_train_acc,
    mean_val_accuracy = mean_val_acc,
    sd_val_accuracy = sd_val_acc,
    mean_val_f1 = mean_val_f1,
    train_accuracy = mean_train_acc,  # For backward compatibility
    test_accuracy = mean_val_acc      # For backward compatibility
  )
  class(model_perf) <- "model_performance"
  
  return(model_perf)
}



 ###### #    #   ##   #      #    #   ##   ##### #  ####  #    #
 #      #    #  #  #  #      #    #  #  #    #   # #    # ##   #
 #####  #    # #    # #      #    # #    #   #   # #    # # #  #
 #      #    # ###### #      #    # ######   #   # #    # #  # #
 #       #  #  #    # #      #    # #    #   #   # #    # #   ##
 ######   ##   #    # ######  ####  #    #   #   #  ####  #    #



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
evaluate_model <- function(model, images_list, image_ids, targets, threshold = cnn_eval_threshold, keep_output = FALSE) {
  
  # Prepare data
  y_labels <- targets[image_ids]
  valid_indices <- !is.na(y_labels)
  
  if (sum(valid_indices) == 0) {
    return(NULL)
  }
  
  X_valid <- images_list[valid_indices]
  y_valid <- as.numeric(y_labels[valid_indices])
  valid_ids <- image_ids[valid_indices]
  
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
  
  # Prediction
  model$eval()
  with_no_grad({
    predictions <- model(X_tensor)
  })
  
  predictions_numeric <- as.numeric(predictions)
  predictions_prob <- 1 / (1 + exp(-predictions_numeric))
  pred_binary <- ifelse(predictions_prob > threshold, 1, 0)
  
  # Metrics
  accuracy <- mean(pred_binary == y_valid)
  
  tp <- sum((pred_binary == 1) & (y_valid == 1))
  tn <- sum((pred_binary == 0) & (y_valid == 0))
  fp <- sum((pred_binary == 1) & (y_valid == 0))
  fn <- sum((pred_binary == 0) & (y_valid == 1))
  
  sensitivity <- if (tp + fn > 0) tp / (tp + fn) else 0
  specificity <- if (tn + fp > 0) tn / (tn + fp) else 0
  precision <- if (tp + fp > 0) tp / (tp + fp) else 0
  f1 <- if ((precision + sensitivity) > 0) {
    2 * (precision * sensitivity) / (precision + sensitivity)
  } else 0
  
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
    accuracy = accuracy,
    raw_outputs = if (keep_output) pred_binary else NULL
  ))
}


 ### #######
  #  #     #
  #  #     #
  #  #     #
  #  #     #
  #  #     #
 ### #######

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