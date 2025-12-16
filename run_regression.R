library(targets)
lapply(list.files("./R", full.names = TRUE), source)

tar_load(ext_data_split)

run_logistic_experiment(ext_data_split, epochs = 2, n_folds = 2)