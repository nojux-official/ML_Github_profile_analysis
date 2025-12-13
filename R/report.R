source("R/constants.R")
library(ggplot2)
library(dplyr)
library(pROC)

# ========== PLOTTING FUNCTIONS ==========

# Plot ROC Curve
plot_roc_curve <- function(actual, predicted_probs, title = "ROC Curve") {
  tryCatch({
    roc_obj <- roc(actual, predicted_probs, 
                   levels = c(0, 1), direction = "<")
    
    plot(roc_obj, main = title, 
         xlab = "False Positive Rate", 
         ylab = "True Positive Rate",
         col = "#2E86AB", lwd = 2)
    
    return(list(auc = roc_obj$auc, roc_obj = roc_obj))
  }, error = function(e) {
    cat("Error plotting ROC curve:", e$message, "\n")
    return(NULL)
  })
}

# Plot Prediction Distribution
plot_prediction_distribution <- function(predictions_df) {
  ggplot(predictions_df, aes(x = probability, fill = factor(prediction))) +
    geom_histogram(bins = 20, alpha = 0.7) +
    facet_wrap(~prediction, scales = "free") +
    scale_fill_manual(values = c("#A23B72", "#2E86AB")) +
    labs(
      title = "Distribution of Predicted Probabilities",
      x = "Probability",
      y = "Frequency",
      fill = "Prediction"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(face = "bold"))
}

# Plot Confusion Matrix
plot_confusion_matrix <- function(actual, predicted) {
  tryCatch({
    library(caret)
    cm <- confusionMatrix(factor(predicted), factor(actual))
    cm_df <- as.data.frame(cm$table)
    
    ggplot(cm_df, aes(x = Prediction, y = Reference, fill = Freq)) +
      geom_tile() +
      geom_text(aes(label = Freq), color = "white", size = 5) +
      scale_fill_gradient(low = "lightblue", high = "#2E86AB") +
      labs(title = "Confusion Matrix") +
      theme_minimal()
  }, error = function(e) {
    cat("Error plotting confusion matrix:", e$message, "\n")
    return(NULL)
  })
}

# Calculate Performance Metrics
calculate_performance_metrics <- function(actual, predicted) {
  tryCatch({
    library(caret)
    cm <- confusionMatrix(factor(predicted), factor(actual))
    
    metrics <- data.frame(
      Metric = c("Accuracy", "Sensitivity (Recall)", "Specificity", 
                 "Precision", "F1-Score", "Balanced Accuracy"),
      Value = c(
        round(cm$overall["Accuracy"], 4),
        round(cm$byClass["Sensitivity"], 4),
        round(cm$byClass["Specificity"], 4),
        round(cm$byClass["Pos Pred Value"], 4),
        round(cm$byClass["F1"], 4),
        round(cm$byClass["Balanced Accuracy"], 4)
      )
    )
    
    return(metrics)
  }, error = function(e) {
    cat("Error calculating metrics:", e$message, "\n")
    return(NULL)
  })
}

# Generate Predictions Summary
get_predictions_summary <- function(predictions_df) {
  data.frame(
    Metric = c("Total Predictions", "Positive (1)", "Negative (0)", 
               "Mean Probability", "Min Probability", "Max Probability", "Std Probability"),
    Value = c(
      nrow(predictions_df),
      sum(predictions_df$prediction, na.rm = TRUE),
      sum(predictions_df$prediction == 0, na.rm = TRUE),
      round(mean(predictions_df$probability, na.rm = TRUE), 4),
      round(min(predictions_df$probability, na.rm = TRUE), 4),
      round(max(predictions_df$probability, na.rm = TRUE), 4),
      round(sd(predictions_df$probability, na.rm = TRUE), 4)
    )
  )
}

