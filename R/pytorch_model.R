# No need for reticulate - using native R torch package

# Convert images to PyTorch tensors
prepare_pytorch_data <- function(image_dir, target_image_size = 7) {
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
    
    # Store as matrix
    images_list[[i]] <- as.matrix(img)[1:target_image_size, 1:target_image_size]
  }
  
  return(list(images = images_list, ids = image_ids))
}

# Build and train PyTorch CNN
train_pytorch_cnn <- function(images_list, image_ids, targets, 
                             epochs = 10, batch_size = 32, learning_rate = 0.001) {
  torch <- import("torch")
  torch.nn <- import("torch.nn")
  torch.optim <- import("torch.optim")
  torch.utils.data <- import("torch.utils.data")
  np <- import("numpy")
  
  # Prepare data
  y_labels <- targets[image_ids]
  valid_indices <- !is.na(y_labels)
  
  X_valid <- images_list[valid_indices]
  y_valid <- as.numeric(y_labels[valid_indices])
  valid_ids <- image_ids[valid_indices]
  
  cat("Training samples:", length(y_valid), "\n")
  
  # Convert to numpy arrays
  X_array <- array(unlist(X_valid), dim = c(length(X_valid), 7, 7))
  X_array <- np$expand_dims(X_array, axis = 1L)  # Add channel dimension
  
  X_tensor <- torch$from_numpy(X_array)$float()
  y_tensor <- torch$from_numpy(np$array(y_valid))$float()$unsqueeze(1L)
  
  # Create dataset
  dataset <- torch.utils.data$TensorDataset(X_tensor, y_tensor)
  dataloader <- torch.utils.data$DataLoader(dataset, batch_size = batch_size, shuffle = TRUE)
  
  # Define CNN model
  model <- nn.Module$new()
  model$initialize <- function(self) {
    super()$`__init__`()
    self$conv1 <- torch.nn$Conv2d(1L, 16L, kernel_size = 3L, padding = 1L)
    self$pool <- torch.nn$MaxPool2d(2L, 2L)
    self$conv2 <- torch.nn$Conv2d(16L, 32L, kernel_size = 3L, padding = 1L)
    self$fc1 <- torch.nn$Linear(32L * 1L * 1L, 64L)
    self$dropout <- torch.nn$Dropout(p = 0.3)
    self$fc2 <- torch.nn$Linear(64L, 1L)
    self$sigmoid <- torch.nn$Sigmoid()
  }
  
  model$forward <- function(self, x) {
    x <- torch.nn.functional$relu(self$conv1(x))
    x <- self$pool(x)
    x <- torch.nn.functional$relu(self$conv2(x))
    x <- self$pool(x)
    x <- x$view(x$size(0L), -1L)
    x <- torch.nn.functional$relu(self$fc1(x))
    x <- self$dropout(x)
    x <- self$sigmoid(self$fc2(x))
    x
  }
  
  model <- nn.Module$new(model)
  
  # Loss and optimizer
  criterion <- torch.nn$BCELoss()
  optimizer <- torch.optim$Adam(model$parameters(), lr = learning_rate)
  
  # Training loop
  for (epoch in 1:epochs) {
    epoch_loss <- 0
    
    for (batch_data in dataloader) {
      X_batch <- batch_data[[1]]
      y_batch <- batch_data[[2]]
      
      # Forward pass
      outputs <- model(X_batch)
      loss <- criterion(outputs, y_batch)
      
      # Backward pass
      optimizer$zero_grad()
      loss$backward()
      optimizer$step()
      
      epoch_loss <- epoch_loss + loss$item()
    }
    
    avg_loss <- epoch_loss / length(dataloader)
    if (epoch %% 2 == 0) {
      cat(sprintf("Epoch %d/%d, Loss: %.4f\n", epoch, epochs, avg_loss))
    }
  }
  
  # Make predictions
  model$eval()
  torch$no_grad()
  
  predictions <- model(X_tensor)$numpy()
  pred_binary <- ifelse(predictions > 0.5, 1, 0)
  
  # Calculate accuracy
  accuracy <- mean(pred_binary == y_valid)
  cat("\nAccuracy:", accuracy, "\n")
  
  results <- data.frame(
    id = valid_ids,
    actual = y_valid,
    predicted = as.numeric(pred_binary),
    probability = as.numeric(predictions)
  )
  
  return(list(
    model = model,
    results = results,
    accuracy = accuracy
  ))
}
