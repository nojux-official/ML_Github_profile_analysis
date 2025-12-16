library(torch)

create_logistic_model <- function(input_dim) {
  LogisticRegression <- nn_module(
    "LogisticRegression",
    initialize = function() {
      self$linear <- nn_linear(input_dim, 1)
    },
    forward = function(x) {
      self$linear(x)
    }
  )
  
  LogisticRegression()
}

train_logistic_model <- function(pca_df, epochs = 100, batch_size = 32, learning_rate = 0.001, weight_decay = 0.01, save_dir = "static", model_name = "logistic_model") {
  feature_cols <- setdiff(colnames(pca_df), c("id", "target"))
  
  pca_df <- pca_df[!is.na(pca_df$target), ]
  
  x_data <- as.matrix(pca_df[, feature_cols])
  y_data <- as.numeric(pca_df$target > 0) # Ensure binary 0/1
  
  x_tensor <- torch_tensor(x_data, dtype = torch_float())
  y_tensor <- torch_tensor(y_data, dtype = torch_float())$unsqueeze(2) # Shape [N, 1]
  
  dataset <- tensor_dataset(x_tensor, y_tensor)
  dataloader <- dataloader(dataset, batch_size = batch_size, shuffle = TRUE)
  
  input_dim <- length(feature_cols)
  model <- create_logistic_model(input_dim)
  
  # Adam optimizer, weight_decay, L2 regularization
  optimizer <- optim_adam(model$parameters, lr = learning_rate, weight_decay = weight_decay)
  criterion <- nn_bce_with_logits_loss()
  
  # Training loop
  cat("Starting Logistic Regression training...\n")
  for (epoch in 1:epochs) {
    model$train()
    total_loss <- 0
    
    coro::loop(for (batch in dataloader) {
      optimizer$zero_grad()
      output <- model(batch[[1]])
      loss <- criterion(output, batch[[2]])
      loss$backward()
      optimizer$step()
      total_loss <- total_loss + loss$item()
    })
    
    if (epoch %% 10 == 0 || epoch == 1) {
      avg_loss <- total_loss / length(dataloader)
      cat(sprintf("Epoch %d/%d: Loss = %.4f\n", epoch, epochs, avg_loss))
    }
  }
  
  # Save model
  if (!dir.exists(save_dir)) dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
  model_path <- file.path(save_dir, paste0(model_name, ".pt"))
  torch_save(model, model_path)
  cat("Model saved to:", model_path, "\n")
  
  # Calculate metrics on training data
  model$eval()
  with_no_grad({
    logits <- model(x_tensor)
    probs <- torch_sigmoid(logits)
  })
  
  preds <- as.numeric(probs > 0.5)
  actuals <- as.numeric(y_data)
  
  accuracy <- mean(preds == actuals)
  
  tp <- sum(preds == 1 & actuals == 1)
  fp <- sum(preds == 1 & actuals == 0)
  fn <- sum(preds == 0 & actuals == 1)
  
  precision <- if (tp + fp > 0) tp / (tp + fp) else 0
  recall <- if (tp + fn > 0) tp / (tp + fn) else 0
  f1 <- if (precision + recall > 0) 2 * precision * recall / (precision + recall) else 0
  
  metrics <- list(
    accuracy = accuracy,
    f1 = f1,
    precision = precision,
    recall = recall
  )
  
  results <- data.frame(
    id = pca_df$id,
    actual = actuals,
    predicted = preds,
    probability = as.numeric(probs)
  )
  
  return(list(
    model_path = model_path,
    metrics = metrics,
    results = results,
    epochs = epochs
  ))
}

predict_logistic <- function(model_info, pca_df) {
  # Handle case where input is the list returned by train_logistic_model
  if (is.list(model_info) && !is.null(model_info$model_path)) {
    model <- torch_load(model_info$model_path)
  } else if (is.character(model_info)) {
    model <- torch_load(model_info)
  } else {
    model <- model_info
  }

  model$eval()
  
  feature_cols <- setdiff(colnames(pca_df), c("id", "target"))
  x_data <- as.matrix(pca_df[, feature_cols])
  x_tensor <- torch_tensor(x_data, dtype = torch_float())
  
  with_no_grad({
    logits <- model(x_tensor)
    probs <- torch_sigmoid(logits)
  })
  
  return(as.numeric(probs))
}
