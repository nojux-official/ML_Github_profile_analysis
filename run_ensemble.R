library(targets)
lapply(list.files("./R", full.names = TRUE), source)

tar_load(test_cnn_model_2)
tar_load(test_cnn_model_4)
tar_load(test_logistic_model_2)
tar_load(test_logistic_model_4)
tar_load(ext_data_split)

generate_and_save_test_predictions(
    list(test_cnn_model_2, test_cnn_model_4),
    list(test_logistic_model_2, test_logistic_model_4), 
    ext_data_split,
    output_file = "static/test_set_predictions.csv"
)
  