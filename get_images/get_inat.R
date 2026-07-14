# Script to download images from iNaturalist
library(rinat)
library(imager)
library(sf)
library(stringr)
library(RCurl)

# Function to download images and save them to files
download_images <- function(spp_recs = NULL, image_folder = "images", spp_folder = NULL) { # nolint
  # Validate inputs
  if (is.null(spp_recs) || is.null(spp_folder)) {
    message("Please set spp_recs and spp_folder")
    return(invisible(NULL))
  }

  # Create directories safely
  dir.create(file.path(image_folder, spp_folder), recursive = TRUE, showWarnings = FALSE) # nolint

  for (image in seq_len(nrow(spp_recs))) {
    message("Processing image number... ", image)

    spp_url <- spp_recs$image_url[image]

    # Remove query parameters safely
    spp_clean <- sub("\\?.*$", "", spp_url)

    message("Original URL: ", spp_url)
    message("Clean URL: ", spp_clean)

    # Only proceed if URL is valid
    if (!is.na(spp_clean) && nzchar(spp_clean)) {
      dest_path <- file.path(image_folder, spp_folder, paste0("spp_", image, ".jpg")) # nolint

      tryCatch(
        {
          download.file(
            url      = spp_clean,
            destfile = dest_path,
            method   = "libcurl",
            mode     = "wb"
          )
        },
        error = function(e) {
          message("Download failed for image ", image, ": ", e$message)
        }
      )
    }
  }
}


# Define geographical region (Great Britain)
#gb_ll <- readRDS("gb.RDS")
gb_ll <- c(49.84, -10.56, 60.85, 1.76) # lon_min, lat_min, lon_max, lat_max

# For testing, only download 50 images for some spp, then downlaod 1000
max_results <- 1000
# American mink
spp_recs1 <- get_inat_obs(
  taxon_name = "Neovison vison",
  bounds = gb_ll,
  maxresults = max_results
)
# European Pine marten
spp_recs2 <- get_inat_obs(
  taxon_name = "Martes martes",
  bounds = gb_ll,
  maxresults = max_results
)
# European otter
spp_recs3 <- get_inat_obs(
  taxon_name = "Lutra lutra",
  bounds = gb_ll,
  maxresults = max_results
)
# European polecat
spp_recs4 <- get_inat_obs(
  taxon_name = "Mustela putorius",
  bounds = gb_ll,
  maxresults = max_results
)
# European weasel
spp_recs5 <- get_inat_obs(
  taxon_name = "Mustela nivalis",
  bounds = gb_ll,
  maxresults = max_results
)
# European stoat
spp_recs6 <- get_inat_obs(
  taxon_name = "Mustela erminea",
  bounds = gb_ll,
  maxresults = max_results
)

download_images(
  spp_recs = spp_recs1,
  spp_folder = "mink"
)
download_images(
  spp_recs = spp_recs2,
  spp_folder = "pine_marten"
)
download_images(
  spp_recs = spp_recs3,
  spp_folder = "otter"
)
download_images(
  spp_recs = spp_recs4,
  spp_folder = "polecat"
)
download_images(
  spp_recs = spp_recs5,
  spp_folder = "weasel"
)
download_images(
  spp_recs = spp_recs6,
  spp_folder = "stoat"
)
