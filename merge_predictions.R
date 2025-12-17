library(dplyr)
library(readr)

# Define file paths
path_predictions <- "static/test_set_predictions.csv"

path_results <- "results_table.csv"

# Check if files exist
if (!file.exists(path_predictions)) {
  stop(paste("File not found:", path_predictions))
}
if (!file.exists(path_results)) {
  stop(paste("File not found:", path_results))
}

# Read the CSV files
message("Reading files...")
df_pred <- read_csv(path_predictions, show_col_types = FALSE)
df_res <- read_csv(path_results, show_col_types = FALSE)

message(paste("Predictions rows:", nrow(df_pred)))
message(paste("Results rows:", nrow(df_res)))

# Exclude MLP_1x128_drop if present
if ("MLP_1x128_drop" %in% names(df_res)) {
  message("Skipping MLP_1x128_drop column...")
  df_res <- df_res %>% select(-MLP_1x128_drop)
}

# Identify join keys
join_keys <- c("id", "target")

# Identify columns in results that are also in predictions (excluding keys)
# We want to merge columns. If columns already exist in predictions, 
# we assume results_table.csv has the authoritative or new data for those columns.
# So we remove them from df_pred before joining to avoid .x/.y suffixes.
cols_in_res <- names(df_res)
cols_to_update <- setdiff(cols_in_res, join_keys)

# Remove columns from df_pred that will be brought in by df_res
df_pred_clean <- df_pred %>% select(-any_of(cols_to_update))

# Perform the merge (full_join to keep all records)
message("Merging dataframes...")
df_merged <- full_join(df_pred_clean, df_res, by = join_keys)

message(paste("Merged rows:", nrow(df_merged)))

# Save the result
message(paste("Saving to", path_predictions))
write_csv(df_merged, path_predictions)

message("Done.")
