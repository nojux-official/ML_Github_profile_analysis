library(EBImage)

# ============================
# Hard-coded feature entries
# ============================
data <- list(
  "0" = c(1574, 3773, 3571, 2672, 2478, 2534, 3129, 3077, 
          1171, 2045, 1539, 902, 1532, 2472, 1122, 2480, 
          3098, 2115, 1578),

  "1" = c(1193, 376, 73, 290, 3129, 1852, 3077, 1171, 
          1022, 2045, 536, 2040, 1533, 1532, 2472, 
          673, 798)
)

# ============================
# Helper: pad to nearest square
# ============================
pad_to_square <- function(vec) {
  n <- length(vec)
  side <- ceiling(sqrt(n))       # nearest square dimension
  new_len <- side * side
  padded <- c(vec, rep(0, new_len - n))
  list(values = padded, side = side)
}

# Output directory
out_dir <- "test_images/"
dir.create(out_dir, showWarnings = FALSE)

# ============================
# Process both entries
# ============================
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

cat("Done.\n")
