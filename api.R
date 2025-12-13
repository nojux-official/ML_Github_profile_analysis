source("R/constants.R")
library(plumber)
library(torch)
library(EBImage)
library(jsonlite)

# Source helper functions
lapply(list.files("./R", full.names = TRUE), source)

# ========== CONFIGURATION ==========
models_dir <- model_save_dir
cache_dir <- api_cache_dir

# ========== ROUTES ==========

#* @apiTitle CNN Image Classification API
#* @apiDescription API for serving CNN model predictions on images

#* Root endpoint
#* @get /
function() {
  list(
    message = "Welcome to the CNN Image Classification API",
    available_endpoints = list(
      models = "/models",
      dataset = "/dataset",
      predict = "/predict",
      report = "/report"
    )
  )
}

#* Serve the performance report
#* @get /report
#* @serializer html list(type="text/html")
function(res) {
  report_path <- file.path("static", "cnn_model_report.html")
  
  if (!file.exists(report_path)) {
    res$status <- 404
    return("<html><body><h1>Report not found</h1><p>Run targets::tar_make(report) to generate it.</p></body></html>")
  }
  
  # Read and return the HTML file
  report_content <- readLines(report_path)
  return(paste(report_content, collapse = "\n"))
}

#* List available CNN models
#* @get /models
function() {
  list(
    models = list.files(models_dir, pattern = "\\.pt$", full.names = FALSE)
  )
}

#* List available datasets
#* @get /dataset
function() {
  list(
    datasets = list.files("dataset", pattern = "\\.(json|csv)$", full.names = FALSE)
  )
}

#* Predict on image features from JSON query
#* @param query:string JSON string with feature arrays
#* @param model:string Model filename (optional, uses latest if not specified)
#* @post /predict
function(query = NULL, model = NULL) {
  
  # Parse query
  if (is.null(query)) {
    return(list(error = "Query parameter required. Provide JSON with feature arrays."))
  }
  
  tryCatch({
    data <- fromJSON(query)
  }, error = function(e) {
    return(list(error = paste("Invalid JSON:", e$message)))
  })
  
  if (is.null(model)) {
    model_files <- list.files(models_dir, pattern = "\\.pt$", full.names = TRUE)
    if (length(model_files) == 0) {
      return(list(error = "No models available"))
    }
    model_path <- model_files[which.max(file.info(model_files)$mtime)]
  } else {
    model_path <- file.path(models_dir, model)
    if (!file.exists(model_path)) {
      return(list(error = paste("Model not found:", model)))
    }
  }
  
  if (!dir.exists(cache_dir)) {
    dir.create(cache_dir, showWarnings = FALSE, recursive = TRUE)
  }
  
  tryCatch({
    apply_transformation(data, out_dir = cache_dir)
  }, error = function(e) {
    return(list(error = paste("Error transforming features:", e$message)))
  })
  
  test_images <- list.files(cache_dir, pattern = "\\.png$", full.names = TRUE)
  
  if (length(test_images) == 0) {
    unlink(cache_dir, recursive = TRUE)
    return(list(error = "No images generated from query"))
  }
  
  tryCatch({
    model <- load_model_from_disk(model_path)
  }, error = function(e) {
    unlink(cache_dir, recursive = TRUE)
    return(list(error = paste("Error loading model:", e$message)))
  })
  
  # Get predictions for all images
  predictions_list <- list()
  
  for (i in seq_along(test_images)) {
    test_image_path <- test_images[i]
    filename <- basename(test_image_path)
    image_id <- gsub("entry_|.png", "", filename)
    
    tryCatch({
      result <- predict_image(model, test_image_path)
      
      predictions_list[[image_id]] <- list(
        image_id = image_id,
        prediction = as.numeric(result$prediction),
        probability = round(as.numeric(result$probability), 4),
        raw_output = round(as.numeric(result$raw_output), 4)
      )
    }, error = function(e) {
      cat("Error predicting on", image_id, ":", e$message, "\n")
    })
  }
  
  # Clean up cache
  unlink(cache_dir, recursive = TRUE)
  
  # Return results
  list(
    model = basename(model_path),
    total_predictions = length(predictions_list),
    predictions = predictions_list
  )
}
