source("R/constants.R")

pad_to_square <- function(vec, target_size = target_image_size) {
  n <- length(vec)
  new_len <- target_size * target_size
  # Truncate if too long, pad if too short
  if (n > new_len) {
    padded <- vec[1:new_len]
  } else {
    padded <- c(vec, rep(0, new_len - n))
  }
  list(values = padded, side = target_size)
}

load_json_features <- function(filepath) {
  json_data <- jsonlite::fromJSON(filepath)
  
  # JSON keys are strings, values are arrays
  data_list <- lapply(json_data, function(x) as.numeric(x))
  
  return(data_list)
}

apply_transformation <- function(data) {
    # out_dir defined in constants
    dir.create(out_dir, showWarnings = FALSE)

    for (id in names(data)) {

        vec <- data[[id]]

        pad <- pad_to_square(vec)
        padded_vec <- pad$values
        side <- pad$side

        # reshape
        img_matrix <- matrix(padded_vec, nrow = side, ncol = side, byrow = TRUE)

        # normalize using max feature = 4000
        img_norm <- img_matrix / 4000

        # create grayscale image
        img <- Image(img_norm)

        # save
        filepath <- sprintf("%s/entry_%s.png", out_dir, id)
        writeImage(img, filepath)

        cat("Saved:", filepath, "(", side, "x", side, ")\n")
    }
}