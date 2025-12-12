#!/bin/bash

# R Package Installation Script
echo "Installing R packages..."

# Set CRAN mirror and install packages using R
R --no-restore --no-save << 'EOF'

# Set CRAN mirror
CRAN_mirror <- "https://cran.r-project.org"

# Install pacman if not already installed
if (!requireNamespace("pacman", quietly = TRUE)) {
  install.packages("pacman", repos = CRAN_mirror)
}

# Load pacman and install all required packages
library(pacman)

# Document processing and utilities
pacman::p_load(rmarkdown, usethis, visNetwork, corrr, DT)

# Machine learning and modeling
pacman::p_load(recipes, rsample, yardstick)

# Data manipulation and visualization (tidyverse)
pacman::p_load(tidyverse)
pacman::p_load(dplyr, ggplot2, tidyr, readr)
pacman::p_load(purrr, tibble, stringr, forcats)

# Programming and workflow
pacman::p_load(rlang, targets, tarchetypes)

# Python integration and deep learning
pacman::p_load(reticulate, tensorflow, keras3)

# Additional useful packages for data science
pacman::p_load(
  # Data manipulation and analysis
  lubridate,     # Date handling
  janitor,       # Data cleaning
  skimr,         # Data summary
  
  # Visualization
  plotly,        # Interactive plots
  patchwork,     # Combining plots
  
  # File I/O
  readxl,        # Excel files
  haven,         # SPSS, Stata, SAS files
  jsonlite,      # JSON handling
  
  # Statistical modeling
  broom,         # Tidy model outputs
  modelr,        # Modeling helpers
  
  # Development tools
  devtools,      # Package development
  roxygen2,      # Documentation
  testthat       # Testing
)

cat("All R packages installed successfully!\n")

EOF

echo "R package installation completed!"