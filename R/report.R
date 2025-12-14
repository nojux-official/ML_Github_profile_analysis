source("R/constants.R")
library(ggplot2)
library(dplyr)
library(pROC)

# Helper to safely extract prediction data
safe_extract_predictions <- function(data) {
  if (is.null(data)) return(NULL)
  
  # If it's a list but not a data frame, try to convert
  if (!is.data.frame(data) && is.list(data)) {
    tryCatch({
      data <- as.data.frame(data)
    }, error = function(e) {
      return(NULL)
    })
  }
  
  if (!is.data.frame(data)) return(NULL)
  
  # Check for minimal required columns
  if (!"prediction" %in% names(data)) return(NULL)
  
  # Handle correct_answer if present, otherwise default to NA
  correct_answer <- if("correct_answer" %in% names(data)) {
    as.numeric(data$correct_answer > 0)
  } else {
    rep(NA, nrow(data))
  }
  
  # Handle raw_output if present
  raw_output <- if("raw_output" %in% names(data)) {
    as.numeric(data$raw_output)
  } else {
    rep(NA, nrow(data))
  }
  
  data.frame(
    image_id = data$image_id,
    prediction = as.numeric(data$prediction),
    probability = as.numeric(data$probability),
    raw_output = raw_output,
    correct_answer = correct_answer,
    stringsAsFactors = FALSE
  )
}

calculate_metrics <- function(actual, predicted) {
  if (is.null(actual) || is.null(predicted)) return(NULL)
  if (length(actual) != length(predicted)) return(NULL)
  if (all(is.na(actual))) return(NULL) # No ground truth
  
  tp <- sum((predicted == 1) & (actual == 1), na.rm = TRUE)
  tn <- sum((predicted == 0) & (actual == 0), na.rm = TRUE)
  fp <- sum((predicted == 1) & (actual == 0), na.rm = TRUE)
  fn <- sum((predicted == 0) & (actual == 1), na.rm = TRUE)
  
  accuracy <- (tp + tn) / (tp + tn + fp + fn)
  sensitivity <- if (tp + fn > 0) tp / (tp + fn) else NA
  specificity <- if (tn + fp > 0) tn / (tn + fp) else NA
  precision <- if (tp + fp > 0) tp / (tp + fp) else NA
  f1 <- if (!is.na(precision) && !is.na(sensitivity) && (precision + sensitivity) > 0) {
    2 * (precision * sensitivity) / (precision + sensitivity)
  } else NA
  balanced_acc <- if (!is.na(sensitivity) && !is.na(specificity)) {
    (sensitivity + specificity) / 2
  } else NA
  
  data.frame(
    Metric = c("Accuracy", "Sensitivity (Recall)", "Specificity", "Precision", "F1-Score", "Balanced Accuracy"),
    Value = c(
      round(accuracy, 4),
      round(sensitivity, 4),
      round(specificity, 4),
      round(precision, 4),
      round(f1, 4),
      round(balanced_acc, 4)
    ),
    stringsAsFactors = FALSE
  )
}

get_roc_stats <- function(actual, predicted_probs) {
  if (is.null(actual) || is.null(predicted_probs)) return(NULL)
  if (all(is.na(actual))) return(NULL)
  if (length(unique(actual)) < 2) return(NULL) # Need both classes
  
  tryCatch({
    # Calculate ROC with CI
    roc_obj <- roc(actual, predicted_probs, levels = c(0, 1), direction = "<", quiet = TRUE, ci = TRUE)
    
    data.frame(
      Metric = c("AUC", "95% CI Lower", "95% CI Upper"),
      Value = c(
        round(as.numeric(roc_obj$auc), 4),
        round(as.numeric(roc_obj$ci[1]), 4),
        round(as.numeric(roc_obj$ci[3]), 4)
      ),
      stringsAsFactors = FALSE
    )
  }, error = function(e) NULL)
}

plot_roc <- function(actual, predicted_probs) {
  if (is.null(actual) || is.null(predicted_probs)) {
    plot.new()
    text(0.5, 0.5, "No data for ROC", cex = 1.5)
    return()
  }
  if (all(is.na(actual)) || length(unique(actual)) < 2) {
    plot.new()
    text(0.5, 0.5, "Insufficient class variance for ROC", cex = 1.5)
    return()
  }
  
  tryCatch({
    roc_obj <- roc(actual, predicted_probs, levels = c(0, 1), direction = "<", quiet = TRUE)
    
    # Use ggroc for better plotting
    g <- ggroc(roc_obj, colour = "#2E86AB", size = 1.2) +
      geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), color = "gray80", linetype = "dashed") +
      labs(
        title = "ROC Curve - CNN Model Performance",
        x = "Specificity",
        y = "Sensitivity"
      ) +
      theme_minimal() +
      theme(plot.title = element_text(face = "bold")) +
      annotate("text", x = 0.25, y = 0.1, 
               label = paste("AUC =", round(as.numeric(roc_obj$auc), 4)), 
               size = 5, color = "#2E86AB")
    
    print(g)
  }, error = function(e) {
    plot.new()
    text(0.5, 0.5, paste("Error plotting ROC:", e$message), cex = 1)
  })
}

plot_confusion_matrix <- function(actual, predicted) {
  if (is.null(actual) || is.null(predicted) || all(is.na(actual))) {
    plot.new()
    text(0.5, 0.5, "No ground truth data for Confusion Matrix", cex = 1.5)
    return()
  }
  
  tp <- sum((predicted == 1) & (actual == 1), na.rm = TRUE)
  tn <- sum((predicted == 0) & (actual == 0), na.rm = TRUE)
  fp <- sum((predicted == 1) & (actual == 0), na.rm = TRUE)
  fn <- sum((predicted == 0) & (actual == 1), na.rm = TRUE)
  
  cm <- matrix(c(tn, fp, fn, tp), nrow = 2, ncol = 2,
               dimnames = list(Predicted = c("Negative", "Positive"),
                              Actual = c("Negative", "Positive")))
  
  plot.new()
  plot.window(xlim = c(0, 3), ylim = c(0, 3))
  
  colors <- matrix(c("#90EE90", "#FFB6C6", "#FFB6C6", "#90EE90"), nrow = 2)
  
  for (i in 1:2) {
    for (j in 1:2) {
      rect(j-0.9, 3.1-i, j-0.1, 3.9-i, col = colors[i,j], border = "black", lwd = 2)
      text(j-0.5, 3.5-i, cm[i,j], cex = 3, font = 2)
    }
  }
  
  text(0.5, 0, "Predicted", cex = 1.2, font = 2)
  text(0, 1.5, "Actual", cex = 1.2, font = 2, srt = 90)
  text(1.5, 3.8, "Negative", cex = 1, pos = 3)
  text(2.5, 3.8, "Positive", cex = 1, pos = 3)
  text(-0.3, 3.5, "Negative", cex = 1, pos = 2)
  text(-0.3, 2.5, "Positive", cex = 1, pos = 2)
}

plot_prediction_distribution <- function(data) {
  if (is.null(data) || !"probability" %in% names(data)) {
    plot.new()
    text(0.5, 0.5, "No data for distribution plot", cex = 1.5)
    return()
  }
  
  ggplot(data, aes(x = probability, fill = factor(prediction))) +
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


