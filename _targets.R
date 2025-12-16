if("pacman" %in% rownames(installed.packages()) == F) install.packages("pacman")
if(!require("targets", quietly = TRUE))
  install.packages("targets")

if(!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
if(!require("EBImage", quietly = TRUE))
  BiocManager::install("EBImage")
if("shiny" %in% rownames(installed.packages()) == F)
  install.packages("https://cran.r-project.org/src/contrib/shiny_1.12.1.tar.gz", repos=NULL, type="source")

if(!require("rmarkdown", quietly = TRUE))
  install.packages("rmarkdown")
if("DET" %in% rownames(installed.packages()) == F)
  install.packages("https://cran.r-project.org/src/contrib/Archive/DET/DET_3.0.1.tar.gz", repos=NULL, type="source")
if("future.apply" %in% rownames(installed.packages()) == F)
  install.packages("https://cran.r-project.org/src/contrib/future.apply_1.20.1.tar.gz", repos=NULL, type="source")
if("crew" %in% rownames(installed.packages()) == F)
  install.packages("crew")

targetPackages <- c("tidyverse", "jsonlite", "tarchetypes", "EBImage",
    "torch", "luz", "torchvision", "torchdatasets", 
    "DT", "httr", "plumber", "shiny",
    "rmarkdown", "DET", "pROC", "ggplot2", "dplyr",
    "lattice", "e1071", "caret", "future.apply", "crew")

pacman::p_load(char = targetPackages)

torch::install_torch()
library(targets)
library(crew)


lapply(list.files("./R", full.names = TRUE), source)
options(tidyverse.quiet = TRUE)
tar_option_set(
  packages = targetPackages,
  controller = crew::crew_controller_local(workers = 8)
)

# Targets pipeline
list(
  tar_target(f_musae_git_target, "dataset/musae_git_target.csv", format = "file"),
  tar_target(f_musae_git_features, "dataset/musae_git_features.json", format = "file"),
  tar_target(f_musae_git_edges, "dataset/musae_git_edges.csv", format = "file"),
  
  tar_target(git_features_data, load_json_features(f_musae_git_features)),
  tar_target(target_labels, load_target_data(f_musae_git_target)),
  tar_target(tabular_df, prepare_tabular(git_features_data, target_labels)),
  tar_target(tabular_pca_df, apply_pca(tabular_df)),
  
  tar_target(git_features_images, apply_transformation(git_features_data)),
  tar_target(image_data, prepare_pytorch_data(git_features_images)),
  
  tar_target(data_split, split_dataset(image_data$images, image_data$ids, target_labels)),
  
  tar_target(cnn_model_1, run_cnn_experiment(data_split, 1)),
  tar_target(cnn_model_2, run_cnn_experiment(data_split, 2)),
  tar_target(cnn_model_4, run_cnn_experiment(data_split, 4)),
  tar_target(cnn_model_5, run_cnn_experiment(data_split, 8)),
  tar_target(cnn_model_10, run_cnn_experiment(data_split, 10)),

  
  tar_target(cnn_models, list(cnn_model_1, cnn_model_2, cnn_model_4, cnn_model_5, cnn_model_10)),
  
  tar_target(single_img_prediction, {
    test_images <- list.files("test_images", pattern = "\\.png$", full.names = TRUE)
    if (length(test_images) > 0) {
      
      #img id
      test_image_path <- test_images[1] # the first img
      filename <- basename(test_image_path)
      image_id <- gsub("entry_|.png", "", filename)
      
      # Load the model with max epochs from disk
      model <- load_model_from_disk("static/cnn_model_2ep.pt")
      
      result <- predict_image(model, test_image_path)
      
      correct_label <- target_labels[image_id]
      
      list(
        test_image = test_image_path,
        image_id = image_id,
        prediction = result$prediction,
        probability = result$probability,
        raw_output = result$raw_output,
        correct_answer = correct_label,
        is_correct = result$prediction == as.numeric(correct_label > 0)
      )
    } else {
      list(test_image = "No test images found", prediction = NA)
    }
    print(paste("Path: ", test_image_path))
    print(paste("Image ID: ", image_id))
    print(paste("Prediction: ", result$prediction))
    print(paste("Probability: ", result$probability))
    print(paste("Raw Output: ", result$raw_output))
    print(paste("Correct Label: ", correct_label))
    print(paste("Is Correct: ", result$prediction == as.numeric(correct_label > 0)))
  }),

  tar_target(single_prediction, {
    query <-  '{
              "0": [1574, 3773, 3571, 2672, 2478, 2534, 3129, 3077, 1171, 2045, 1539, 902, 1532, 2472, 1122, 2480, 3098, 2115, 1578],
              "1": [1193, 376, 73, 290, 3129, 1852, 3077, 1171, 1022, 2045, 536, 2040, 1533, 1532, 2472, 673, 798],
              "2": [1574, 3773, 925, 1728, 2815, 2963, 3077, 364, 1171, 536, 1867, 2472, 1122, 2532, 664, 28, 3311, 1768, 869],
              "3": [3964, 3773, 4003, 928, 1852, 3077, 364, 1022, 3763, 2045, 3859, 3771, 234, 664, 703]
              }'
    data <- jsonlite::fromJSON(query)

    if (!dir.exists("cache/")) {
      dir.create("cache/", showWarnings = FALSE, recursive = TRUE)
    }
    apply_transformation(data, out_dir = "cache/")
    test_images <- list.files("cache", pattern = "\\.png$", full.names = TRUE)
    
    if (length(test_images) > 0) {
      # Load the model with max epochs from disk
      model <- load_model_from_disk("static/cnn_model_2ep.pt")
      
      predictions_list <- list()
      
      for (i in seq_along(test_images)) {
        test_image_path <- test_images[i]
        filename <- basename(test_image_path)
        image_id <- gsub("entry_|.png", "", filename)
        
        result <- predict_image(model, test_image_path, threshold = cnn_eval_threshold)
        
        # Look up correct label if available
        correct_label <- NA
        if (image_id %in% names(target_labels)) {
          correct_label <- as.numeric(target_labels[image_id] > 0)
        }
        
        predictions_list[[i]] <- data.frame(
          image_path = test_image_path,
          image_id = image_id,
          prediction = result$prediction,
          probability = result$probability,
          raw_output = result$raw_output,
          correct_answer = correct_label,
          stringsAsFactors = FALSE
        )
      }
      
      all_predictions <- do.call(rbind, predictions_list)
      rownames(all_predictions) <- NULL
      
      print(all_predictions)
      
      all_predictions
    } else {
      data.frame(test_image = "No test images found", prediction = NA)
    }
    cache_files <- list.files("cache/", pattern = "\\.png$", full.names = TRUE)
    if (length(cache_files) > 0) {
      file.remove(cache_files)
    }
  }),
  tar_render(report, "report.Rmd", output_file = "static/cnn_model_report.html")
)
