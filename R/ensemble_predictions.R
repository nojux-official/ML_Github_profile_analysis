source("R/constants.R")
library(torch)

# Generate and save test set predictions for all models
# This function expects lists of evaluation results (from evaluate_model/evaluate_logistic_model)
# which contain 'results' dataframes with 'id', 'actual', 'predicted', 'probability'
generate_and_save_test_predictions <- function(cnn_results_list, logistic_results_list, test_split, output_file = "static/test_set_predictions.csv") {
  
  cat(strrep("=", 70), "\n")
  cat("AGGREGATING TEST SET PREDICTIONS\n")
  cat(strrep("=", 70), "\n\n")
  
  # Start with the test set IDs and targets from the split
  # This ensures we have all samples even if some models missed some (though they shouldn't)
  test_ids <- test_split$test$ids
  test_targets <- test_split$test$targets
  
  results_df <- data.frame(
    id = as.character(test_ids),
    target = as.numeric(test_targets[test_ids]),
    stringsAsFactors = FALSE
  )
  
  # Filter to valid targets if needed, or keep all
  # The evaluation functions filter to valid targets.
  results_df <- results_df[!is.na(results_df$target), ]
  
  cat("Test set size:", nrow(results_df), "samples\n\n")
  
  # Process CNN results
  for (i in seq_along(cnn_results_list)) {
    res <- cnn_results_list[[i]]
    model_name <- paste0("CNN_Model_", i)
    
    if (!is.null(res) && !is.null(res$results)) {
      # Extract ID and Probability
      model_preds <- res$results[, c("id", "probability")]
      colnames(model_preds)[2] <- model_name
      
      results_df <- merge(results_df, model_preds, by = "id", all.x = TRUE)
      cat("  ✓ Added", model_name, "\n")
    }
  }
  
  # Process Logistic results
  for (i in seq_along(logistic_results_list)) {
    res <- logistic_results_list[[i]]
    model_name <- paste0("Logistic_Model_", i)
    
    if (!is.null(res) && !is.null(res$results)) {
      # Extract ID and Probability
      model_preds <- res$results[, c("id", "probability")]
      colnames(model_preds)[2] <- model_name
      
      results_df <- merge(results_df, model_preds, by = "id", all.x = TRUE)
      cat("  ✓ Added", model_name, "\n")
    }
  }
  
  # Sort by id
  results_df <- results_df[order(as.numeric(results_df$id)), ]
  rownames(results_df) <- NULL
  
  # Create output directory if needed
  output_dir <- dirname(output_file)
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  }
  
  # Save to file
  write.csv(results_df, file = output_file, row.names = FALSE)
  
  cat("\n✓ Test predictions saved to:", output_file, "\n")
  cat("  Dimensions:", nrow(results_df), "samples x", ncol(results_df), "columns\n\n")
  
  return(results_df)
}


# Collect test run summaries from model performance objects
collect_test_summaries <- function(cnn_models, logistic_models) {
  
  cat(strrep("=", 70), "\n")
  cat("TEST RUN SUMMARIES\n")
  cat(strrep("=", 70), "\n\n")
  
  summaries <- list()
  
  # CNN model summaries
  cat("CNN MODELS:\n")
  cat(strrep("-", 70), "\n")
  for (i in seq_along(cnn_models)) {
    model_perf <- cnn_models[[i]]
    epochs <- model_perf$epochs
    model_name <- paste0("CNN_", epochs, "ep")
    
    cat("Model:", model_name, "\n")
    cat("  Mean Validation Accuracy:", round(model_perf$mean_val_accuracy, 4), "±", 
        round(model_perf$sd_val_accuracy, 4), "\n")
    cat("  Mean Validation F1:", round(model_perf$mean_val_f1, 4), "\n")
    cat("  Fold Accuracies:", paste(round(model_perf$fold_accuracies, 4), collapse = ", "), "\n")
    cat("  CV Models:", length(model_perf$cv_models), "folds\n\n")
    
    summaries[[model_name]] <- data.frame(
      model = model_name,
      type = "CNN",
      epochs = epochs,
      mean_accuracy = model_perf$mean_val_accuracy,
      sd_accuracy = model_perf$sd_val_accuracy,
      mean_f1 = model_perf$mean_val_f1,
      n_folds = model_perf$n_folds
    )
  }
  
  # Logistic model summaries
  cat("LOGISTIC MODELS:\n")
  cat(strrep("-", 70), "\n")
  for (i in seq_along(logistic_models)) {
    model_perf <- logistic_models[[i]]
    epochs <- model_perf$epochs
    model_name <- paste0("Logistic_", epochs, "ep")
    
    cat("Model:", model_name, "\n")
    cat("  Mean Validation Accuracy:", round(model_perf$mean_val_accuracy, 4), "±", 
        round(model_perf$sd_val_accuracy, 4), "\n")
    cat("  Mean Validation F1:", round(model_perf$mean_val_f1, 4), "\n")
    cat("  Fold Accuracies:", paste(round(model_perf$fold_accuracies, 4), collapse = ", "), "\n")
    cat("  CV Models:", length(model_perf$cv_models), "folds\n\n")
    
    summaries[[model_name]] <- data.frame(
      model = model_name,
      type = "Logistic",
      epochs = epochs,
      mean_accuracy = model_perf$mean_val_accuracy,
      sd_accuracy = model_perf$sd_val_accuracy,
      mean_f1 = model_perf$mean_val_f1,
      n_folds = model_perf$n_folds
    )
  }
  
  # Combine into single summary dataframe
  summary_df <- do.call(rbind, summaries)
  rownames(summary_df) <- NULL
  
  cat(strrep("=", 70), "\n")
  cat("SUMMARY TABLE\n")
  cat(strrep("=", 70), "\n")
  print(summary_df)
  cat("\n")
  
  return(summary_df)
}

# Save comprehensive test output report
save_test_report <- function(cnn_models, logistic_models, test_predictions, 
                            output_dir = "static", report_name = "test_report") {
  
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  }
  
  # Collect summaries
  summaries <- collect_test_summaries(cnn_models, logistic_models)
  
  # Save summaries
  summary_file <- file.path(output_dir, paste0(report_name, "_summary.csv"))
  write.csv(summaries, file = summary_file, row.names = FALSE)
  cat("✓ Summary saved to:", summary_file, "\n")
  
  # Save predictions (already saved by generate_test_predictions, but include path info)
  predictions_file <- file.path(output_dir, "test_set_predictions.csv")
  cat("✓ Test predictions available at:", predictions_file, "\n")
  
  # Create detailed text report
  report_file <- file.path(output_dir, paste0(report_name, ".txt"))
  sink(report_file)
  
  cat("TEST RUN REPORT\n")
  cat("Generated:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n\n")
  
  cat("CNN MODELS PERFORMANCE\n")
  cat(strrep("=", 70), "\n\n")
  for (i in seq_along(cnn_models)) {
    model_perf <- cnn_models[[i]]
    epochs <- model_perf$epochs
    cat("CNN Model:", epochs, "epochs\n")
    cat("  Mean Validation Accuracy:", round(model_perf$mean_val_accuracy, 4), "±", 
        round(model_perf$sd_val_accuracy, 4), "\n")
    cat("  Mean Validation F1:", round(model_perf$mean_val_f1, 4), "\n")
    cat("  Individual Fold Accuracies:\n")
    for (f in 1:length(model_perf$fold_accuracies)) {
      cat("    Fold", f, ":", round(model_perf$fold_accuracies[f], 4), "\n")
    }
    cat("\n")
  }
  
  cat("LOGISTIC MODELS PERFORMANCE\n")
  cat(strrep("=", 70), "\n\n")
  for (i in seq_along(logistic_models)) {
    model_perf <- logistic_models[[i]]
    epochs <- model_perf$epochs
    cat("Logistic Model:", epochs, "epochs\n")
    cat("  Mean Validation Accuracy:", round(model_perf$mean_val_accuracy, 4), "±", 
        round(model_perf$sd_val_accuracy, 4), "\n")
    cat("  Mean Validation F1:", round(model_perf$mean_val_f1, 4), "\n")
    cat("  Individual Fold Accuracies:\n")
    for (f in 1:length(model_perf$fold_accuracies)) {
      cat("    Fold", f, ":", round(model_perf$fold_accuracies[f], 4), "\n")
    }
    cat("\n")
  }
  
  cat("TEST SET PREDICTIONS\n")
  cat(strrep("=", 70), "\n\n")
  cat("Total Test Samples:", nrow(test_predictions), "\n")
  cat("Prediction Columns:", paste(colnames(test_predictions), collapse = ", "), "\n\n")
  
  cat("First 10 Test Predictions:\n")
  print(head(test_predictions, 10))
  
  sink()
  cat("✓ Detailed report saved to:", report_file, "\n")
  
  return(list(
    summary = summaries,
    report_file = report_file,
    predictions_file = predictions_file
  ))
}
