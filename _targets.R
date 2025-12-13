if("pacman" %in% rownames(installed.packages()) == F) install.packages("pacman")
if(!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
if(!require("EBImage", quietly = TRUE))
  BiocManager::install("EBImage")
if(!require("targets", quietly = TRUE))
  install.packages("targets")

targetPackages <- c("tidyr", "jsonlite", "torch", "luz", "torchvision", "torchdatasets")
pacman::p_load(char = targetPackages)

torch::install_torch()
library(targets)


lapply(list.files("./R", full.names = TRUE), source)
options(tidyverse.quiet = TRUE)
tar_option_set(packages = targetPackages)

# Targets pipeline
list(
  tar_target(f_musae_git_target, "dataset/musae_git_target.csv", format = "file"),
  tar_target(f_musae_git_features, "dataset/musae_git_features.json", format = "file"),
  tar_target(f_musae_git_edges, "dataset/musae_git_edges.csv", format = "file"),
  
  tar_target(git_features_data, load_json_features(f_musae_git_features)),
  
  tar_target(git_features_images, apply_transformation(git_features_data)),
  
  tar_target(target_labels, load_target_data(f_musae_git_target)),
  
  tar_target(image_data, prepare_pytorch_data(git_features_images)),
  
  tar_target(cnn_model, train_pytorch_cnn(
    image_data$images,
    image_data$ids,
    target_labels,
    epochs = 2,
    batch_size = 16,
    learning_rate = 0.001
  ))
)
