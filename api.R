source("R/constants.R")
library(plumber)
library(torch)
library(EBImage)
library(jsonlite)

lapply(list.files("./R", full.names = TRUE), source)

# ========== CONFIGURATION ==========
models_dir <- model_save_dir
cache_dir <- api_cache_dir

# ========== ROUTES ==========

#* @apiTitle CNN Image Classification API
#* @apiDescription API for serving CNN model predictions on images

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

#* @get /report
#* @serializer html list(type="text/html")
function(res) {
  report_path <- file.path("static", "cnn_model_report.html")
  
  if (!file.exists(report_path)) {
    res$status <- 404
    return("<html><body><h1>Report not found</h1><p>Run targets::tar_make(report) to generate it.</p></body></html>")
  }
  
  report_content <- readLines(report_path)
  return(paste(report_content, collapse = "\n"))
}

#* @get /models
function() {
  list(
    models = list.files(models_dir, pattern = "\\.pt$", full.names = FALSE),
    download_url = "/models/<model_name>"
  )
}

#* @get /models/<model_name>
#* @serializer contentType list(type="application/octet-stream")
function(model_name, res) {
  model_path <- file.path(models_dir, model_name)
  
  if (!file.exists(model_path)) {
    res$status <- 404
    return(list(error = paste("Model not found:", model_name)))
  }
  
  if (!grepl("\\.pt$", model_name)) {
    res$status <- 400
    return(list(error = "Only .pt files can be downloaded"))
  }
  
  res$headers[["Content-Disposition"]] <- paste0("attachment; filename=", model_name)
  res$headers[["Content-Type"]] <- "application/octet-stream"
  
  readBin(model_path, "raw", n = file.size(model_path))
}

#* @get /dataset
function() {
  list(
    datasets = list.files("dataset", pattern = "\\.(json|csv)$", full.names = FALSE)
  )
}

#* @param query:string JSON string with feature arrays
#* @param model:string Model filename (optional, uses latest if not specified)
#* @post /predict
function(query = NULL, model = NULL) {
  
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
  
  unlink(cache_dir, recursive = TRUE)
  
  list(
    model = basename(model_path),
    total_predictions = length(predictions_list),
    predictions = predictions_list
  )
}
