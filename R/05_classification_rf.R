# ============================================================
# puno_carbon_stock_pipeline.R
# Full R pipeline for carbon stock estimation in high-Andean
# ecosystems, Puno, Peru — Sentinel-2 + Random Forest + MERESE/MVC
# ============================================================
# Author: Paulo César Calla Chambi
# Steps:
#   1. Download Sentinel-2 imagery (Copernicus Data Space)
#   2. Preprocessing (SCL cloud masking + seasonal median composite)
#   3. Spectral indices (NDVI, NDWI-Gao, EVI)
#   4. Predictor stack (Sentinel-2 bands + indices + DEM derivatives)
#   5. Random Forest land cover classification
#   6. Carbon stock valuation under MERESE/MVC price scenarios
# ============================================================

library(terra)
library(sf)
library(randomForest)
library(caret)
library(httr2)   # for Copernicus OAuth2 authentication

# ============================================================
# STEP 1 — Download Sentinel-2 imagery
# ============================================================
# Copernicus Data Space requires OAuth2 authentication.
# Store credentials as environment variables — never hardcode
# them in the script (especially in a public repo).
#
#   Sys.setenv(CDSE_CLIENT_ID = "your_client_id")
#   Sys.setenv(CDSE_CLIENT_SECRET = "your_client_secret")

get_cdse_token <- function() {
  resp <- request("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token") |>
    req_body_form(
      grant_type    = "client_credentials",
      client_id     = Sys.getenv("CDSE_CLIENT_ID"),
      client_secret = Sys.getenv("CDSE_CLIENT_SECRET")
    ) |>
    req_perform()
  resp_body_json(resp)$access_token
}

# Study area: Puno department, covered by 17 Sentinel-2 tiles
tile_ids <- c("19LDH", "19LDJ")  # replace with your full list of 17 tiles

download_sentinel2_tile <- function(tile_id, date_start, date_end, token, dest_dir) {
  # Placeholder for the actual download call against the
  # Copernicus Data Space OData/STAC API for a given tile
  # and date range. Kept generic here — fill in with your
  # specific query parameters (cloud cover threshold, product
  # type S2MSI2A, etc.)
  message(sprintf("Downloading tile %s (%s to %s)...", tile_id, date_start, date_end))
}

dir.create("data/raw", recursive = TRUE, showWarnings = FALSE)
token <- get_cdse_token()
for (tile in tile_ids) {
  download_sentinel2_tile(tile, "2024-01-01", "2025-12-31", token, "data/raw")
}

# ============================================================
# STEP 2 — Preprocessing: SCL cloud masking + seasonal composite
# ============================================================
mask_clouds_scl <- function(img, scl_band) {
  # SCL classes to keep: 4 (vegetation), 5 (bare soil), 6 (water),
  # 7 (unclassified), 11 (snow) — mask out 3,8,9,10 (clouds/shadow/cirrus)
  valid_classes <- c(4, 5, 6, 7, 11)
  cloud_mask <- scl_band %in% valid_classes
  mask(img, cloud_mask, maskvalue = FALSE)
}

build_seasonal_composite <- function(tile_files, scl_files) {
  imgs <- lapply(seq_along(tile_files), function(i) {
    img <- rast(tile_files[i])
    scl <- rast(scl_files[i])
    mask_clouds_scl(img, scl)
  })
  img_stack <- do.call(c, imgs)
  # Median composite per band across the time series
  app(img_stack, median, na.rm = TRUE)
}

# Example call — replace file lists with your actual downloaded scenes
# seasonal_composite <- build_seasonal_composite(tile_files, scl_files)
# writeRaster(seasonal_composite, "data/processed/seasonal_composite.tif", overwrite = TRUE)

seasonal_composite <- rast("data/processed/seasonal_composite.tif")

# ============================================================
# STEP 3 — Spectral indices: NDVI, NDWI (Gao), EVI
# ============================================================
# Band naming convention assumed: B2 (blue), B4 (red), B8 (NIR), B11 (SWIR1)

ndvi <- (seasonal_composite$B8 - seasonal_composite$B4) /
        (seasonal_composite$B8 + seasonal_composite$B4)

ndwi_gao <- (seasonal_composite$B8 - seasonal_composite$B11) /
            (seasonal_composite$B8 + seasonal_composite$B11)

evi <- 2.5 * (seasonal_composite$B8 - seasonal_composite$B4) /
       (seasonal_composite$B8 + 6 * seasonal_composite$B4 -
        7.5 * seasonal_composite$B2 + 1)

names(ndvi) <- "NDVI"
names(ndwi_gao) <- "NDWI"
names(evi) <- "EVI"

spectral_indices <- c(ndvi, ndwi_gao, evi)
writeRaster(spectral_indices, "data/processed/spectral_indices.tif", overwrite = TRUE)

# ============================================================
# STEP 4 — Predictor stack: Sentinel-2 bands + indices + DEM
# ============================================================
dem <- rast("data/raw/copernicus_dem_puno.tif")
dem <- resample(dem, seasonal_composite, method = "bilinear")

slope   <- terrain(dem, v = "slope", unit = "degrees")
aspect  <- terrain(dem, v = "aspect", unit = "degrees")
twi_num <- log((terrain(dem, v = "flowdir") + 1) / tan(slope * pi / 180 + 0.001))
names(twi_num) <- "TWI"

predictor_stack <- c(
  seasonal_composite[[c("B2", "B3", "B4", "B8", "B11", "B12")]],
  spectral_indices,
  dem, slope, aspect, twi_num
)
names(predictor_stack)[names(predictor_stack) == "copernicus_dem_puno"] <- "elevation"

# Confirm 13 layers as expected
stopifnot(nlyr(predictor_stack) == 13)

writeRaster(predictor_stack, "data/processed/predictor_stack_13var.tif", overwrite = TRUE)

# ============================================================
# STEP 5 — Random Forest land cover classification
# ============================================================
training_polygons <- st_read("data/training/training_polygons.shp")

# Expected class counts (from field digitizing log):
# bofedal=56, pajonal=50, queñual=52, suelo_desnudo=52, agua=62
table(training_polygons$class)

training_points <- st_sample(
  training_polygons,
  size = rep(150, nrow(training_polygons)),
  type = "regular"
)

training_sf <- st_sf(
  class = training_polygons$class[
    st_nearest_feature(training_points, training_polygons)
  ],
  geometry = training_points
)

extracted_values <- terra::extract(predictor_stack, vect(training_sf))
training_data <- cbind(class = training_sf$class, extracted_values[, -1])
training_data$class <- as.factor(training_data$class)
training_data <- na.omit(training_data)

set.seed(123)
train_index <- caret::createDataPartition(training_data$class, p = 0.7, list = FALSE)
train_set <- training_data[train_index, ]
test_set  <- training_data[-train_index, ]

rf_model <- randomForest(
  class ~ .,
  data       = train_set,
  ntree      = 500,
  mtry       = floor(sqrt(ncol(train_set) - 1)),
  importance = TRUE
)
print(rf_model)

varImpPlot(rf_model, main = "Predictor importance — RF classification")
write.csv(as.data.frame(importance(rf_model)), "outputs/variable_importance.csv")

predictions <- predict(rf_model, test_set)
confusion <- caret::confusionMatrix(predictions, test_set$class)
print(confusion)
write.csv(as.data.frame(confusion$table), "outputs/confusion_matrix.csv")

classified_map <- terra::predict(predictor_stack, rf_model, type = "response", na.rm = TRUE)
writeRaster(classified_map, "outputs/landcover_classified_puno.tif", overwrite = TRUE)

cell_area_ha <- prod(res(classified_map)) / 10000
area_by_class <- freq(classified_map) |> as.data.frame() |>
  transform(area_ha = count * cell_area_ha)
write.csv(area_by_class, "outputs/area_by_class_ha.csv")
print(area_by_class)

# ============================================================
# STEP 6 — Carbon stock valuation (MERESE/MVC scenarios)
# ============================================================
# Replace these carbon density values (t C / ha) with the
# literature-derived figures used in your manuscript.
carbon_density <- data.frame(
  class      = c("bofedal", "pajonal", "queñual", "suelo_desnudo", "agua"),
  t_C_per_ha = c(NA, NA, NA, 0, 0)   # <-- fill in real values
)

carbon_stock <- merge(area_by_class, carbon_density, by = "class")
carbon_stock$total_tC <- carbon_stock$area_ha * carbon_stock$t_C_per_ha

# Carbon price scenarios (USD per tCO2e) — adjust to your MVC scenarios
co2_per_c <- 3.67  # conversion factor C -> CO2
price_scenarios <- c(low = 5, medium = 15, high = 30)

for (scenario in names(price_scenarios)) {
  col_name <- paste0("value_usd_", scenario)
  carbon_stock[[col_name]] <- carbon_stock$total_tC * co2_per_c * price_scenarios[scenario]
}

write.csv(carbon_stock, "outputs/carbon_stock_valuation.csv", row.names = FALSE)
print(carbon_stock)

# ============================================================
# End of pipeline
# ============================================================
