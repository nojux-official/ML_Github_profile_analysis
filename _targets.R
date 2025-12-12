# Prepare environment and load required libraries
if("pacman" %in% rownames(installed.packages()) == F) install.packages("pacman")
if(!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
if(!require("EBImage", quietly = TRUE))
  BiocManager::install("EBImage")

# apt-get update && apt-get install -y libfftw3-dev libglpk40

# Install torch packages for R
targetPackages <- c("tidyr", "jsonlite", "torch", "luz", "torchvision", "torchdatasets")
pacman::p_load(char = targetPackages)
library(targets)

# Load your R files / conflicted
lapply(list.files("./R", full.names = TRUE), source)
options(tidyverse.quiet = TRUE)
tar_option_set(packages = targetPackages)

# Targets pipeline
list(
  tar_target(f_musae_git_target, "dataset/musae_git_target.csv", format = "file"),
  tar_target(f_musae_git_features, "dataset/musae_git_features.json", format = "file"),
  tar_target(f_musae_git_edges, "dataset/musae_git_edges.csv", format = "file"),
  
  tar_target(git_features_data, load_json_features(f_musae_git_features)),
  
  tar_target(git_features_images, apply_transformation(git_features_data))
  
  
)
