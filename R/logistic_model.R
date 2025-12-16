source("R/constants.R")
library(torch)
library(luz)



   ##   #####   ####  #    # # ##### ######  ####  ##### #    # #####  ######
  #  #  #    # #    # #    # #   #   #      #    #   #   #    # #    # #
 #    # #    # #      ###### #   #   #####  #        #   #    # #    # #####
 ###### #####  #      #    # #   #   #      #        #   #    # #####  #
 #    # #   #  #    # #    # #   #   #      #    #   #   #    # #   #  #
 #    # #    #  ####  #    # #   #   ######  ####    #    ####  #    # ######


# Logistic Regression model for tabular feature classification
create_logistic_model <- function(input_dim) {
  nn_module(
    "LogisticRegression",
    initialize = function() {
      self$linear <- nn_linear(input_dim, 1)
      nn_init_xavier_uniform_(self$linear$weight)
      nn_init_constant_(self$linear$bias, 0)
    },
    forward = function(x) {
      self$linear(x)
    }
  )
}



 ##### #####    ##   # #    #
   #   #    #  #  #  # ##   #
   #   #    # #    # # # #  #
   #   #####  ###### # #  # #
   #   #   #  #    # # #   ##
   #   #    # #    # # #    #

# Train logistic model on a single fold
train_logistic_model <- function(tabular_df, epochs = 100, batch_size = 32, learning_rate = 0.001, 
                                 weight_decay = 0.01, threshold = 0.5, model_name = "logistic_model", 
                                 save_dir = model_save_dir) {
  feature_cols <- setdiff(colnames(tabular_df), c("id", "target"))
  
  # Remove rows with missing target
  tabular_df <- tabular_df[!is.na(tabular_df$target), ]
  
  x_data <- as.matrix(tabular_df[, feature_cols])
  y_data <- as.numeric(tabular_df$target > 0) # Ensure binary 0/1
  
  # Standardize input data
  mean_val <- colMeans(x_data)
  sd_val <- apply(x_data, 2, sd)
  sd_val[sd_val < 1e-6] <- 1  # Avoid division by zero
  x_data_scaled <- sweep(sweep(x_data, 2, mean_val, "-"), 2, sd_val, "/")
  
  x_tensor <- torch_tensor(x_data_scaled, dtype = torch_float())
  y_tensor <- torch_tensor(y_data, dtype = torch_float())$unsqueeze(2) # Shape [N, 1]
  
  # Create dataset
  dataset <- torch::dataset(
    name = "tabular_dataset",
    initialize = function(x, y) {
      self$x <- x
      self$y <- y
    },
    .getitem = function(index) {
      list(x = self$x[index, ], y = self$y[index, ])
    },
    .length = function() {
      self$x$shape[1]
    }
  )
  
  train_ds <- dataset(x_tensor, y_tensor)
  train_dl <- torch::dataloader(train_ds, batch_size = batch_size, shuffle = TRUE)
  
  input_dim <- length(feature_cols)
  model <- create_logistic_model(input_dim)
  
  # Loss function with class weighting for imbalanced data
  n_pos <- sum(y_data == 1)
  n_neg <- sum(y_data == 0)
  pos_weight_val <- if(n_pos > 0) n_neg / n_pos else 1
  loss_fn <- nn_bce_with_logits_loss(pos_weight = torch_tensor(pos_weight_val))
  
  # Using luz to train the model
  cat("Starting Logistic Regression training...\n")
  fitted_model <- model %>%
    setup(
      loss = loss_fn,
      optimizer = optim_adam
    ) %>%
    set_opt_hparams(lr = learning_rate, weight_decay = weight_decay) %>%
    fit(
      data = train_dl,
      epochs = epochs,
      verbose = FALSE
    )
  
  trained_model <- fitted_model$model
  
  # Make predictions on full dataset
  trained_model$eval()
  with_no_grad({
    logits <- trained_model(x_tensor)
    probs <- torch_sigmoid(logits)
  })
  
  predictions_numeric <- as.numeric(probs)
  predictions_prob <- predictions_numeric
  pred_binary <- ifelse(predictions_prob > threshold, 1, 0)
  
  # Calculate metrics
  accuracy <- mean(pred_binary == y_data)
  
  tp <- sum((pred_binary == 1) & (y_data == 1))
  tn <- sum((pred_binary == 0) & (y_data == 0))
  fp <- sum((pred_binary == 1) & (y_data == 0))
  fn <- sum((pred_binary == 0) & (y_data == 1))
  
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
    id = tabular_df$id,
    actual = y_data,
    predicted = pred_binary,
    probability = predictions_prob
  )
  
  save_logistic_model_to_disk(trained_model, save_dir, model_name)
  
  return(list(
    model = trained_model,
    results = results,
    metrics = metrics,
    accuracy = accuracy,
    fitted = fitted_model
  ))
}



 ###### #    #   ##   #      #    #   ##   ##### #  ####  #    #
 #      #    #  #  #  #      #    #  #  #    #   # #    # ##   #
 #####  #    # #    # #      #    # #    #   #   # #    # # #  #
 #      #    # ###### #      #    # ######   #   # #    # #  # #
 #       #  #  #    # #      #    # #    #   #   # #    # #   ##
 ######   ##   #    # ######  ####  #    #   #   #  ####  #    #



# Run logistic regression experiment with 5-fold cross-validation
run_logistic_experiment <- function(data_split, epochs = 100, batch_size = 32, learning_rate = 0.001, 
                                    threshold = 0.5, n_folds = 5) {
  model_name <- paste0("logistic_model_", epochs, "ep")
  
  # Combine train and test data for k-fold CV
  all_data <- data_split$train$pca_df
  
  # Create fold indices
  n_samples <- nrow(all_data)
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
    
    fold_train_data <- all_data[train_indices, ]
    fold_val_data <- all_data[val_indices, ]
    
    cat("Train samples:", nrow(fold_train_data), "| Validation samples:", nrow(fold_val_data), "\n")
    
    # Train model on this fold
    fold_model_name <- paste0(model_name, "_fold", fold)
    fold_train_result <- train_logistic_model(
      tabular_df = fold_train_data,
      epochs = epochs,
      batch_size = batch_size,
      learning_rate = learning_rate,
      threshold = threshold,
      model_name = fold_model_name
    )
    
    # Evaluate on validation set
    fold_val_result <- evaluate_logistic_model(
      model = fold_train_result$model,
      tabular_df = fold_val_data,
      threshold = threshold
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
  
  # Save the best model (highest validation accuracy) as the final model
  best_fold <- which.max(all_val_accuracies)
  best_model <- cv_models[[best_fold]]
  save_logistic_model_to_disk(best_model, model_save_dir, model_name)
  cat("Saved best model from fold", best_fold, "with accuracy:", round(all_val_accuracies[best_fold], 4), "\n\n")
  
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



# Predict on a single data point
predict_logistic <- function(model, tabular_data) {
  model$eval()
  
  feature_cols <- setdiff(colnames(tabular_data), c("id", "target"))
  x_data <- as.matrix(tabular_data[, feature_cols])
  
  # Standardize using training data stats (in production, use stored training stats)
  mean_val <- colMeans(x_data)
  sd_val <- apply(x_data, 2, sd)
  sd_val[sd_val < 1e-6] <- 1
  x_data_scaled <- sweep(sweep(x_data, 2, mean_val, "-"), 2, sd_val, "/")
  
  x_tensor <- torch_tensor(x_data_scaled, dtype = torch_float())
  
  with_no_grad({
    logits <- model(x_tensor)
    probs <- torch_sigmoid(logits)
  })
  
  return(as.numeric(probs))
}

# Evaluate logistic model on a dataset
evaluate_logistic_model <- function(model, tabular_df, threshold = logistic_eval_threshold, keep_output = FALSE) {
  
  # Prepare data
  tabular_df <- tabular_df[!is.na(tabular_df$target), ]
  
  if (nrow(tabular_df) == 0) {
    return(NULL)
  }
  
  feature_cols <- setdiff(colnames(tabular_df), c("id", "target"))
  x_data <- as.matrix(tabular_df[, feature_cols])
  y_data <- as.numeric(tabular_df$target > 0)
  
  # Standardize input data
  mean_val <- colMeans(x_data)
  sd_val <- apply(x_data, 2, sd)
  sd_val[sd_val < 1e-6] <- 1
  x_data_scaled <- sweep(sweep(x_data, 2, mean_val, "-"), 2, sd_val, "/")
  
  x_tensor <- torch_tensor(x_data_scaled, dtype = torch_float())
  
  # Prediction
  model$eval()
  with_no_grad({
    logits <- model(x_tensor)
    probs <- torch_sigmoid(logits)
  })
  
  predictions_numeric <- as.numeric(probs)
  predictions_prob <- predictions_numeric
  pred_binary <- ifelse(predictions_prob > threshold, 1, 0)
  
  # Metrics
  accuracy <- mean(pred_binary == y_data)
  
  tp <- sum((pred_binary == 1) & (y_data == 1))
  tn <- sum((pred_binary == 0) & (y_data == 0))
  fp <- sum((pred_binary == 1) & (y_data == 0))
  fn <- sum((pred_binary == 0) & (y_data == 1))
  
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
    id = tabular_df$id,
    actual = y_data,
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

save_logistic_model_to_disk <- function(model, save_dir = model_save_dir, model_name = "logistic_model") {
  if (!dir.exists(save_dir)) {
    dir.create(save_dir, showWarnings = FALSE, recursive = TRUE)
  }
  
  model_path <- file.path(save_dir, paste0(model_name, ".pt"))
  torch_save(model, model_path)
  
  cat("Model saved to:", model_path, "\n")
  
  return(model_path)
}

load_logistic_model_from_disk <- function(model_path = "static/logistic_model.pt") {
  if (!file.exists(model_path)) {
    stop("Model file not found at: ", model_path)
  }
  
  model <- torch_load(model_path)
  cat("Model loaded from:", model_path, "\n")
  
  return(model)
}
