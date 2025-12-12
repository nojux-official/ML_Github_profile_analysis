# Prepare environment and load required libraries
if("pacman" %in% rownames(installed.packages()) == F) install.packages("pacman")
pacman::p_load(utf8, targets, tarchetypes, rmarkdown, dotenv, conflicted, pROC, doParallel)
pacman::p_load(bs4Dash, gt, DT, pingr, shiny, shinybusy, shinycssloaders, shinyWidgets, visNetwork)
if("DET" %in% rownames(installed.packages()) == F)
  install.packages("https://cran.r-project.org/src/contrib/Archive/DET/DET_3.0.1.tar.gz", repos=NULL, type="source")


# apt-get update && apt-get install -y libfftw3-dev

# Install BiocManager and EBImage (Bioconductor package)
if(!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
if(!require("EBImage", quietly = TRUE))
  BiocManager::install("EBImage")

targetPackages <- c("tidyr", "jsonlite")
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
