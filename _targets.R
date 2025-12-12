# Prepare environment and load required libraries
if("pacman" %in% rownames(installed.packages()) == F) install.packages("pacman")
pacman::p_load(utf8, targets, tarchetypes, rmarkdown, dotenv, conflicted, pROC, doParallel)
pacman::p_load(bs4Dash, gt, DT, pingr, shiny, shinybusy, shinycssloaders, shinyWidgets, visNetwork)
if("DET" %in% rownames(installed.packages()) == F)
  install.packages("https://cran.r-project.org/src/contrib/Archive/DET/DET_3.0.1.tar.gz", repos=NULL, type="source")
targetPackages <- c("readxl", "parallel", "pbapply", "ggplot2", "plotly", "dplyr","reshape2",
                    "tidytext","tidyr","textclean","lexicon", "tm", "slam", "topicmodels",
                    "lda", "text2vec", "glmnetUtils", "DET", "caret", "e1071", "PresenceAbsence", "tuneRanger", "mlr")
pacman::p_load(char = targetPackages)
library(targets)
library(tuneRanger)
library(mlr)


# Set main variables
minDocs <- 11 # discard rare terms
minChars <- 2 # minimum word length
numTopics <- 10 # number of topics to extract
numWords <- 7 # best terms for topic profiling
ctrlDTM <- list(wordLengths=c(minChars,Inf),
                bounds=list(global=c(minDocs,Inf)))
ctrlSC <- list(language="lt")
ctrlCTM <- list(verbose=10)
plotCols <- c("shop","date","rating","bad")

# Load your R files / conflicted
lapply(list.files("./R", full.names = TRUE), source)
options(tidyverse.quiet = TRUE)
tar_option_set(packages = targetPackages)

# Targets pipeline
list(
  tar_target(raw_data_file, "data/LT_atsiliepimai_2011-2018.xlsx", format = "file"),
  tar_target(stop_words_file, "data/LT_stopwords_iso_sulotynintas.txt", format = "file"),
  # Data overview
  tar_target(stop_words, read.table(stop_words_file)[,1]),
  tar_target(data_frame, clean_text(readxl::read_excel(raw_data_file))),
  tar_target(top_shops, sort(table(data_frame$shop))),
  tar_target(data_overview, plot_data_overview(data_frame[,plotCols], top_shops)),
  # Topic modeling
  tar_target(text_corpus, tm::SimpleCorpus(tm::DataframeSource(data_frame), control=ctrlSC)),
  tar_target(text_matrix, tm::DocumentTermMatrix(text_corpus, control=c(list(stopwords=stop_words),ctrlDTM))),
  tar_target(selected_idx, slam::row_sums(text_matrix)>0),
  tar_target(topic_model, topicmodels::CTM(text_matrix[selected_idx,], k=numTopics, control=ctrlCTM)),
  tar_target(model_post, topicmodels::posterior(topic_model)),
  tar_target(topic_names, apply(lda::top.topic.words(model_post$terms, numWords, by.score=T), 2, paste, collapse = " ")),
  tar_target(topic_results, plot_topic_results(model_post$topics, topic_names, data_frame[selected_idx,'bad'])),
  # Machine learning
tar_target(train_split, split(data_frame, f=factor(data_frame$shop!="pigu.lt"))),
tar_target(train_iter, text2vec::itoken(train_split[["TRUE"]]$text)),
tar_target(test_iter, text2vec::itoken(train_split[["FALSE"]]$text)),
tar_target(vectorizer, text2vec_vectorizer(train_iter, minDocs, stop_words)),
tar_target(lsa_topics, c(32, 64, 96, 128)),

tar_target(model_output_long, train_test_LDA_logit(train_iter, test_iter, vectorizer, train_split[["TRUE"]]$bad, train_split[["FALSE"]]$bad, lsa_topics), pattern = map(lsa_topics)),

tar_target(rf_num_trees, 256),
tar_target(
  rf_model_output_long,
  train_test_rf_model(
    train_iter,
    test_iter,
    vectorizer,
    train_split[["TRUE"]]$bad,
    train_split[["FALSE"]]$bad,
    num_trees = rf_num_trees
  )
),

tar_target(combined_model_output_long, dplyr::bind_rows(model_output_long, rf_model_output_long)),
tar_target(model_output_pivot, tidyr::spread(combined_model_output_long, model, score)),
  tar_target(model_performance, plot_detection_results(model_output_pivot)),
  tar_target(model_EER_results, sapply(model_performance@detCurves, function(x) x@eer)),
  # tar_target(model_confusions, calc_confusion_matrix(model_output_pivot,names(model_EER_results))),
  tar_target(best_model_name, names(which.min(model_EER_results))),
  tar_target(best_model_dim, as.numeric(strsplit(best_model_name,"D_")[[1]][1])),
  tar_target(final_iter, text2vec::itoken(data_frame$text)),
  tar_target(vectorizer_final, text2vec_vectorizer(final_iter, minDocs, stop_words)),
  # tar_target(model_otuput_files, train_test_lda_logit(final_iter, tstIter=NULL, vectorizer_final, data_frame$bad, tstTarget=NULL, best_model_dim), format = "file"),
  # Markdown compilation
  tar_render(report, "report.Rmd")
)
