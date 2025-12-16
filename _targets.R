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

  tar_target(logistic_model, train_logistic_model(tabular_pca_df)),
  
  tar_target(model_results, list(cnn_model_1, cnn_model_2, cnn_model_4, cnn_model_5, cnn_model_10)),
  

  
  tar_render(report, "report.Rmd", output_file = "static/cnn_model_report.html")
)
