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
  
  tar_target(ext_data_split, split_dataset(image_data$images, image_data$ids, target_labels,
                                         tabular_df, tabular_pca_df,      
                                         split_ratio = 0.8, seed = 123)),
  
  tar_target(cnn_model_2, run_cnn_experiment(ext_data_split, 2)),
  tar_target(cnn_model_4, run_cnn_experiment(ext_data_split, 4)),
  tar_target(cnn_model_8, run_cnn_experiment(ext_data_split, 8)),

  tar_target(logistic_model_2, run_logistic_experiment(ext_data_split, epochs = 2)),

  tar_target(test_cnn_model_2, {
      model <- load_model_from_disk(
          file.path('static', paste(cnn_model_2$model_name, '.pt', sep="")))
      evaluate_model(model, ext_data_split$test$images,
          ext_data_split$test$ids, ext_data_split$test$targets, keep_output = TRUE)
    }
  ),
  tar_target(test_cnn_model_4, {
      model <- load_model_from_disk(
          file.path('static', paste(cnn_model_4$model_name, '.pt', sep="")))
      evaluate_model(model, ext_data_split$test$images,
          ext_data_split$test$ids, ext_data_split$test$targets, keep_output = TRUE)
    }
  ),
  tar_target(test_cnn_model_8, {
      model <- load_model_from_disk(
          file.path('static', paste(cnn_model_8$model_name, '.pt', sep="")))
      evaluate_model(model, ext_data_split$test$images,
          ext_data_split$test$ids, ext_data_split$test$targets, keep_output = TRUE)
    }
  ),


  tar_target(test_logistic_model_2, {
      model <- load_model_from_disk(
          file.path('static', paste(logistic_model_2$model_name, '.pt', sep="")))
      evaluate_logistic_model(model, ext_data_split$test$pca_df, keep_output = TRUE)
    }
  )

  
  # tar_target(
  #   test_predictions,
  #   generate_and_save_test_predictions(
  #     list(cnn_model_2, cnn_model_4, cnn_model_8),
  #     logistic_model,
  #     ext_data_split,
  #     output_file = "static/test_set_predictions.csv"
  #   )
  # )
  
  # tar_render(report, "report.Rmd", output_file = "static/cnn_model_report.html")
)
