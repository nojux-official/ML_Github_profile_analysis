source("R/constants.R")
library(torch)
library(luz)

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
    img <- EBImage::resize(img, w = target_image_size, h = target_image_size)
    
    # Extract filename (ID)
    filename <- basename(png_files[i])
    image_ids[i] <- gsub("entry_|.png", "", filename)
    
    # Store as matrix (normalized)
    img_matrix <- as.matrix(img)[1:target_image_size, 1:target_image_size]
    images_list[[i]] <- img_matrix
  }
  
  return(list(images = images_list, ids = image_ids))
}

# Build and train CNN using native R torch
train_pytorch_cnn <- function(images_list, image_ids, targets, 
                             epochs = 10, batch_size = 32, learning_rate = 0.001) {
  
  # Prepare data
  y_labels <- targets[image_ids]
  valid_indices <- !is.na(y_labels)
  
  X_valid <- images_list[valid_indices]
  y_valid <- as.numeric(y_labels[valid_indices])
  valid_ids <- image_ids[valid_indices]
  
  cat("Training samples:", length(y_valid), "\n")
  
  # Convert to tensor format (N, 1, H, W)
  n_samples <- length(X_valid)
  h <- nrow(X_valid[[1]])
  w <- ncol(X_valid[[1]])
  
  X_array <- array(0, dim = c(n_samples, 1, h, w))
  for (i in seq_along(X_valid)) {
    X_array[i, 1, , ] <- X_valid[[i]]
  }
  
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
  
  # Define CNN model
  net <- nn_module(
    "CNN",
    initialize = function() {
      self$conv1 <- nn_conv2d(1, 16, kernel_size = 3, padding = 1)
      self$pool <- nn_max_pool2d(2, 2)
      self$conv2 <- nn_conv2d(16, 32, kernel_size = 3, padding = 1)
      
      # Calculate flattened size after pooling
      # Input: (H, W) -> after conv1+pool: (H/2, W/2) -> after conv2+pool: (H/4, W/4)
      flat_size <- 32 * (h %/% 4) * (w %/% 4)
      
      self$fc1 <- nn_linear(flat_size, 64)
      self$dropout <- nn_dropout(p = 0.3)
      self$fc2 <- nn_linear(64, 1)
    },
    forward = function(x) {
      x %>%
        self$conv1() %>%
        nnf_relu() %>%
        self$pool() %>%
        self$conv2() %>%
        nnf_relu() %>%
        self$pool() %>%
        torch_flatten(start_dim = 2) %>%
        self$fc1() %>%
        nnf_relu() %>%
        self$dropout() %>%
        self$fc2() %>%
        torch_sigmoid()
    }
  )
  
  # Setup and train model using luz
  fitted_model <- net %>%
    setup(
      loss = nn_bce_loss(),
      optimizer = optim_adam,
      metrics = list(
        luz_metric_mse()
      )
    ) %>%
    fit(
      data = train_dl,
      epochs = epochs,
      verbose = TRUE
    )
  
  # Get trained model
  model <- fitted_model$model
  
  # Make predictions on full dataset
  model$eval()
  with_no_grad({
    predictions <- model(X_tensor)
  })
  
  predictions_numeric <- as.numeric(predictions)
  pred_binary <- ifelse(predictions_numeric > 0.5, 1, 0)
  
  # Calculate accuracy
  accuracy <- mean(pred_binary == y_valid)
  cat("\nAccuracy:", accuracy, "\n")
  
  results <- data.frame(
    id = valid_ids,
    actual = y_valid,
    predicted = pred_binary,
    probability = predictions_numeric
  )
  
  return(list(
    model = model,
    results = results,
    accuracy = accuracy,
    fitted = fitted_model
  ))
}
