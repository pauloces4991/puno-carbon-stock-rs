# ============================================================
# 05_classification_rf.R
# Random Forest land cover classification for carbon stock
# mapping in high-Andean ecosystems, Puno, Peru
# ============================================================
# Input : 13-variable predictor stack (Sentinel-2 bands,
#         spectral indices, DEM derivatives) + training polygons
# Output: classified land cover raster (5 classes) + accuracy report
# ============================================================

library(terra)
library(randomForest)
library(sf)
library(caret)

# ---- 1. Load predictor stack -------------------------------
# Replace with the path to the stack produced in 04_predictor_stack.R
predictor_stack <- rast("data/processed/predictor_stack_13var.tif")

# Expected layer names — adjust to match your actual stack
# e.g. c("B2","B3","B4","B8","B11","B12","NDVI","NDWI","EVI",
#        "elevation","slope","aspect","TWI")
names(predictor_stack)

# ---- 2. Load training polygons -----------------------------
# One shapefile per class, or a single file with a "class" field
training_polygons <- st_read("data/training/training_polygons.shp")

# Class counts (for reference / QA against your digitizing log)
table(training_polygons$class)
# Expected: bofedal=56, pajonal=50, queñual=52,
#           suelo_desnudo=52, agua=62

# ---- 3. Extract predictor values at training locations -----
training_points <- st_sample(
  training_polygons,
  size = rep(150, nrow(training_polygons)), # points per polygon
  type = "regular"
)

training_sf <- st_sf(
  class = training_polygons$class[
    st_nearest_feature(training_points, training_polygons)
  ],
  geometry = training_points
)

extracted_values <- terra::extract(
  predictor_stack,
  vect(training_sf)
)

training_data <- cbind(class = training_sf$class, extracted_values[,-1])
training_data$class <- as.factor(training_data$class)
training_data <- na.omit(training_data)

# ---- 4. Train/test split ------------------------------------
set.seed(123)
train_index <- caret::createDataPartition(
  training_data$class, p = 0.7, list = FALSE
)
train_set <- training_data[train_index, ]
test_set  <- training_data[-train_index, ]

# ---- 5. Fit Random Forest model ------------------------------
rf_model <- randomForest(
  class ~ .,
  data       = train_set,
  ntree      = 500,
  mtry       = floor(sqrt(ncol(train_set) - 1)),
  importance = TRUE
)

print(rf_model)

# ---- 6. Variable importance -----------------------------------
varImpPlot(rf_model, main = "Predictor importance — RF classification")
importance_df <- as.data.frame(importance(rf_model))
write.csv(importance_df, "outputs/variable_importance.csv")

# ---- 7. Accuracy assessment on held-out test set ---------------
predictions <- predict(rf_model, test_set)
confusion <- caret::confusionMatrix(predictions, test_set$class)

print(confusion)
# Overall Accuracy and Kappa are reported in confusion$overall
# Per-class Producer's/User's Accuracy in confusion$byClass

write.csv(
  as.data.frame(confusion$table),
  "outputs/confusion_matrix.csv"
)

# ---- 8. Classify the full raster stack --------------------------
classified_map <- terra::predict(
  predictor_stack,
  rf_model,
  type = "response",
  na.rm = TRUE
)

writeRaster(
  classified_map,
  "outputs/landcover_classified_puno.tif",
  overwrite = TRUE
)

# ---- 9. Compute classified area per class (hectares) ------------
cell_area_ha <- prod(res(classified_map)) / 10000

area_by_class <- freq(classified_map) |>
  as.data.frame() |>
  transform(area_ha = count * cell_area_ha)

write.csv(area_by_class, "outputs/area_by_class_ha.csv")
print(area_by_class)

# ============================================================
# Next step (06_carbon_valuation.R):
# multiply area_by_class$area_ha by literature-derived carbon
# density values (t C / ha) per class to estimate total carbon
# stock, then apply MERESE/MVC price scenarios.
# ============================================================
