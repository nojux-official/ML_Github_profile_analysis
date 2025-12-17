source("R/constants.R")
library(ggplot2)
library(dplyr)
library(pROC)
library(DET)

# Helper to convert model_performance to DETs object for plotting
create_dets_object <- function(model_performance_list) {
  # Handle single model or list of models
  if (inherits(model_performance_list, "model_performance")) {
    model_list <- list(model_performance_list)
  } else if (is.list(model_performance_list)) {
    model_list <- model_performance_list
  } else {
    return(NULL)
  }
  
  # Prepare data for detc
  responses <- list()
  predictors_list <- list()
  names_list <- c()
  
  for (i in seq_along(model_list)) {
    model <- model_list[[i]]
    
    # Handle different result structures
    results_df <- if (!is.null(model$results)) model$results else model$test_results
    
    if (!is.null(results_df) && length(unique(results_df$actual)) > 1) {
      responses[[i]] <- factor(results_df$actual, levels = c(0, 1))
      predictors_list[[i]] <- results_df$probability
      
      # Determine model name
      if (!is.null(model$name)) {
        names_list[i] <- model$name
      } else if (!is.null(model$epochs)) {
        names_list[i] <- paste0("Epochs_", model$epochs)
      } else {
        names_list[i] <- paste0("Model_", i)
      }
    }
  }
  
  if (length(responses) == 0) return(NULL)
  
  # Combine all predictors into a matrix
  predictors_matrix <- do.call(cbind, predictors_list)
  colnames(predictors_matrix) <- names_list
  
  # Create DETs object
  dets_obj <- detc(
    response = responses[[1]],
    predictors = predictors_matrix,
    names = names_list,
    positive = "1"
  )
  
  return(dets_obj)
}

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
  # Ensure factors with same levels
  u_levels <- union(unique(actual), unique(predicted))
  u_levels <- sort(u_levels)
  
  # Force 0 and 1 if possible
  if(all(u_levels %in% c(0, 1))) {
    levels <- c(0, 1)
  } else {
    levels <- u_levels
  }
  
  act_fac <- factor(actual, levels = levels)
  pred_fac <- factor(predicted, levels = levels)
  
  tbl <- table(Predicted = pred_fac, Actual = act_fac)
  
  # Extract counts
  # Assuming levels are 0, 1
  # tbl[row, col] -> tbl[Predicted, Actual]
  tn <- tbl[1, 1] # Pred 0, Act 0
  fn <- tbl[1, 2] # Pred 0, Act 1
  fp <- tbl[2, 1] # Pred 1, Act 0
  tp <- tbl[2, 2] # Pred 1, Act 1
  
  # Row sums (Predicted totals)
  row_sum_0 <- tn + fn
  row_sum_1 <- fp + tp
  
  # Col sums (Actual totals)
  col_sum_0 <- tn + fp
  col_sum_1 <- fn + tp
  
  total <- sum(tbl)
  
  # Metrics
  # User Accuracy (Precision-like) - Row wise
  user_acc_0 <- if(row_sum_0 > 0) tn / row_sum_0 else 0 # NPV
  user_acc_1 <- if(row_sum_1 > 0) tp / row_sum_1 else 0 # Precision
  
  # Producer Accuracy (Recall-like) - Col wise
  prod_acc_0 <- if(col_sum_0 > 0) tn / col_sum_0 else 0 # Specificity
  prod_acc_1 <- if(col_sum_1 > 0) tp / col_sum_1 else 0 # Recall
  
  overall_acc <- (tp + tn) / total
  
  # Kappa
  pe <- ((row_sum_0 * col_sum_0) + (row_sum_1 * col_sum_1)) / (total^2)
  kappa <- if(1 - pe != 0) (overall_acc - pe) / (1 - pe) else 0
  
  # Create data frame for ggplot
  plot_data <- data.frame(
    x = integer(), y = integer(), 
    label = character(), 
    fill = character(),
    stringsAsFactors = FALSE
  )
  
  # Helper to add cell
  add_cell <- function(x, y, label, fill) {
    data.frame(x = x, y = y, label = as.character(label), fill = fill, stringsAsFactors = FALSE)
  }
  
  # Colors
  col_class1 <- "#FFFACD" # LemonChiffon
  col_class2 <- "#FFE4B5" # Moccasin
  col_total <- "#98FB98" # PaleGreen
  col_metric <- "#FF6347" # Tomato
  col_dark_green <- "#2E8B57"
  col_gray <- "gray90"
  
  # Row 1 (Pred 0) - Y=4
  plot_data <- rbind(plot_data, add_cell(1, 4, tn, col_class2))
  plot_data <- rbind(plot_data, add_cell(2, 4, fn, col_class1))
  plot_data <- rbind(plot_data, add_cell(3, 4, row_sum_0, col_total))
  plot_data <- rbind(plot_data, add_cell(4, 4, sprintf("%.1f%%", user_acc_0*100), col_metric))
  
  # Row 2 (Pred 1) - Y=3
  plot_data <- rbind(plot_data, add_cell(1, 3, fp, col_class1))
  plot_data <- rbind(plot_data, add_cell(2, 3, tp, col_class2))
  plot_data <- rbind(plot_data, add_cell(3, 3, row_sum_1, col_total))
  plot_data <- rbind(plot_data, add_cell(4, 3, sprintf("%.1f%%", user_acc_1*100), col_metric))
  
  # Row 3 (Total Truth) - Y=2
  plot_data <- rbind(plot_data, add_cell(1, 2, col_sum_0, col_total))
  plot_data <- rbind(plot_data, add_cell(2, 2, col_sum_1, col_total))
  plot_data <- rbind(plot_data, add_cell(3, 2, total, col_dark_green))
  plot_data <- rbind(plot_data, add_cell(4, 2, "", "gray50"))
  
  # Row 4 (Producer Acc) - Y=1
  plot_data <- rbind(plot_data, add_cell(1, 1, sprintf("%.1f%%", prod_acc_0*100), col_metric))
  plot_data <- rbind(plot_data, add_cell(2, 1, sprintf("%.1f%%", prod_acc_1*100), col_metric))
  plot_data <- rbind(plot_data, add_cell(3, 1, "", "gray50"))
  plot_data <- rbind(plot_data, add_cell(4, 1, "", "gray50"))
  
  # Plot
  ggplot(plot_data, aes(x = x, y = y, fill = fill)) +
    geom_tile(color = "white", size = 1) +
    geom_text(aes(label = label), size = 5, fontface = "bold") +
    scale_fill_identity() +
    scale_x_continuous(
      breaks = 1:4, 
      labels = c("Class 0", "Class 1", "Total", "User Acc"),
      position = "top",
      expand = c(0, 0)
    ) +
    scale_y_continuous(
      breaks = 1:4, 
      labels = c("Prod Acc", "Total", "Class 1", "Class 0"),
      expand = c(0, 0)
    ) +
    labs(
      title = "Confusion Matrix",
      subtitle = sprintf("Overall Accuracy: %.1f%%  |  Kappa: %.3f", overall_acc*100, kappa),
      x = "Actual Class",
      y = "Predicted Class"
    ) +
    theme_minimal() +
    theme(
      axis.text = element_text(size = 12, face = "bold"),
      axis.title = element_text(size = 14, face = "bold"),
      panel.grid = element_blank(),
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 14, hjust = 0.5)
    )
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

plot_det <- function(actual, predicted_probs) {
  if (is.null(actual) || is.null(predicted_probs)) {
    plot.new()
    text(0.5, 0.5, "No data for DET plot", cex = 1.5)
    return(NULL)
  }
  if (all(is.na(actual)) || length(unique(actual)) < 2) {
    plot.new()
    text(0.5, 0.5, "Insufficient class variance for DET plot", cex = 1.5)
    return(NULL)
  }
  
  tryCatch({
    # Calculate DET curve manually using different thresholds
    thresholds <- seq(0, 1, by = 0.01)
    det_data <- data.frame()
    
    for (threshold in thresholds) {
      pred_binary <- ifelse(predicted_probs >= threshold, 1, 0)
      
      # Calculate FNR and FPR
      fn <- sum((pred_binary == 0) & (actual == 1), na.rm = TRUE)
      tp <- sum((pred_binary == 1) & (actual == 1), na.rm = TRUE)
      fp <- sum((pred_binary == 1) & (actual == 0), na.rm = TRUE)
      tn <- sum((pred_binary == 0) & (actual == 0), na.rm = TRUE)
      
      # Miss Rate (False Negative Rate) and False Alarm Rate (False Positive Rate)
      fnr <- if ((tp + fn) > 0) fn / (tp + fn) else NA
      fpr <- if ((fp + tn) > 0) fp / (fp + tn) else NA
      
      det_data <- rbind(det_data, data.frame(
        Threshold = threshold,
        FNR = fnr * 100,
        FPR = fpr * 100
      ))
    }
    
    # Remove NA values
    det_data <- det_data[!is.na(det_data$FNR) & !is.na(det_data$FPR), ]
    
    # Create DET plot on log scale
    p <- ggplot(det_data, aes(x = FPR, y = FNR)) +
      geom_line(color = "#2E86AB", size = 1.2) +
      geom_point(color = "#2E86AB", size = 2, alpha = 0.6) +
      scale_x_log10(limits = c(0.1, 50)) +
      scale_y_log10(limits = c(0.1, 50)) +
      labs(
        title = "DET Curve - Model Performance",
        x = "False Alarm Rate (%)",
        y = "Miss Rate (%)"
      ) +
      theme_minimal() +
      theme(
        plot.title = element_text(face = "bold", size = 12),
        panel.grid.major = element_line(color = "gray80"),
        panel.grid.minor = element_line(color = "gray90", linetype = "dotted")
      )
    
    return(p)
  }, error = function(e) {
    warning(paste("Error creating DET plot:", e$message))
    return(NULL)
  })
}

