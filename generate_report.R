#!/usr/local/bin/Rscript

# Report generation script
if(!require("rmarkdown", quietly = TRUE)) install.packages("rmarkdown")
if(!require("pROC", quietly = TRUE)) install.packages("pROC")
if("DET" %in% rownames(installed.packages()) == F)
  install.packages("https://cran.r-project.org/src/contrib/Archive/DET/DET_3.0.1.tar.gz", repos=NULL, type="source")
if(!require("ggplot2", quietly = TRUE)) install.packages("ggplot2")

library(rmarkdown)

cat("Generating CNN Model Performance Report...\n")

# Render the report
render(
  "report.Rmd",
  output_format = "html_document",
  output_file = "cnn_model_report.html",
  output_dir = ".",
  quiet = FALSE
)

cat("\n✓ Report generated successfully!\n")
cat("Output: cnn_model_report.html\n")
