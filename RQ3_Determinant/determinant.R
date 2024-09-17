library(Hmisc)
library(rms)
library(dplyr)
library(randomForest)
library(caret)
library(e1071)
library(pROC)
library(ScottKnottESD)
library(tidyr)
library(jsonlite)

file_path <- "D:/Research/RQ2/classifier/data.csv"
data <- read.csv(file_path, check.names = FALSE)

target <- data[['quality']]

# Remove unnecessary columns
data <- data %>% select(-modelId, -quality)

# Calculate the correlation matrix
corr_matrix <- cor(data, method = "spearman")

# Identify feature pairs with a correlation greater than 0.7
high_corr_pairs <- which(abs(corr_matrix) > 0.7 & abs(corr_matrix) <= 1, arr.ind = TRUE)
high_corr_pairs <- high_corr_pairs[high_corr_pairs[,1] < high_corr_pairs[,2],]

# Output feature pairs with high correlation
high_corr_features <- apply(high_corr_pairs, 1, function(idx) {
  paste(colnames(data)[idx[1]], "-", colnames(data)[idx[2]], ":", round(corr_matrix[idx[1], idx[2]], 2))
})

cat("Highly correlated feature pairs (>0.7):\n")
cat(high_corr_features, sep = "\n")

# Perform hierarchical clustering
hc <- hclust(as.dist(1 - abs(corr_matrix)), method = "complete")

# Function to plot dendrogram
plot_dendrogram <- function(hc, title) {
  plot(hc, main = title, xlab = "", sub = "", cex = 0.6)
}

# Plot feature clustering
plot_dendrogram(hc, "Feature Clustering")

# Remove highly correlated features
reduced_data <- data %>% select(-`since-create`, -`since-last-update`, -`since-last-model-update`, -`has-license`, -`num-heading`)

# Perform redundancy analysis and remove redundant features
perform_redundancy_analysis <- function(data) {
  data_df <- as.data.frame(data)
  redun_res <- redun(~ ., data = data_df, nk = 0)
  redundant_features <- redun_res$Out
  cat("Redundant features:\n")
  print(redundant_features)
  data_reduced <- data_df %>% select(-one_of(redundant_features))
  return(data_reduced)
}

# Apply redundancy analysis
reduced_data <- perform_redundancy_analysis(reduced_data)

cat("Remaining features:\n")
print(colnames(reduced_data))

# Add target variable back to the data
reduced_data$quality <- target

# Replace hyphens with dots in column names
colnames(reduced_data) <- gsub("-", ".", colnames(reduced_data))

# Ensure the target variable is a factor
reduced_data$quality <- as.factor(reduced_data$quality)

all_feature_importance <- data.frame()
all_metrics <- data.frame()

# Run 10 iterations with 10-fold cross-validation
for (run in 1:10) {
  folds <- createFolds(reduced_data$quality, k = 10, list = TRUE, returnTrain = TRUE)
  
  for (i in 1:length(folds)) {
    train_index <- folds[[i]]
    train_data <- reduced_data[train_index, ]
    test_data <- reduced_data[-train_index, ]
    
    # Train random forest model
    rf_model <- randomForest(quality ~ ., data = train_data, ntree = 100)
    
    # Calculate feature importance
    feature_importance <- importance(rf_model)
    feature_importance <- data.frame(Feature = rownames(feature_importance), Importance = feature_importance[, 1])
    feature_importance$Fold <- i
    feature_importance$Run <- run
    all_feature_importance <- rbind(all_feature_importance, feature_importance)
    
    # Predict and calculate performance metrics
    predictions <- predict(rf_model, newdata = test_data)
    prob_predictions <- predict(rf_model, newdata = test_data, type = "prob")[, 2]
    
    confusion_matrix <- confusionMatrix(predictions, test_data$quality)
    accuracy <- confusion_matrix$overall['Accuracy']
    precision <- confusion_matrix$byClass['Pos Pred Value']
    recall <- confusion_matrix$byClass['Sensitivity']
    f1 <- 2 * ((precision * recall) / (precision + recall))
    roc_curve <- roc(test_data$quality, prob_predictions)
    auc <- auc(roc_curve)
    
    # Store metrics
    fold_metrics <- data.frame(
      Run = run,
      Fold = i,
      Accuracy = accuracy,
      Precision = precision,
      Recall = recall,
      F1 = f1,
      AUC = auc
    )
    all_metrics <- rbind(all_metrics, fold_metrics)
  }
}

# Calculate average feature importance
average_feature_importance <- all_feature_importance %>%
  group_by(Feature) %>%
  summarise(Importance = mean(Importance))

# Calculate average performance metrics
average_metrics <- all_metrics %>%
  summarise(
    Accuracy = mean(Accuracy),
    Precision = mean(Precision),
    Recall = mean(Recall),
    F1 = mean(F1),
    AUC = mean(AUC)
  )

# Print feature importance
cat("Feature Importance:\n")
print(average_feature_importance)

print(average_metrics)

# Sort feature importance
average_feature_importance <- average_feature_importance %>%
  arrange(desc(Importance))

# Sort all feature importance by average importance
all_feature_importance <- all_feature_importance %>%
  mutate(Feature = factor(Feature, levels = average_feature_importance$Feature)) %>%
  arrange(Feature) %>%
  select(Feature, Fold, Importance)

# Convert feature importance to wide format for Scott-Knott ESD test
data_wide <- all_feature_importance %>%
  pivot_wider(names_from = Fold, values_from = Importance, values_fn = mean)

# Perform Scott-Knott ESD test
data_wide_t <- as.data.frame(t(as.matrix(data_wide[, -1])))
colnames(data_wide_t) <- data_wide$Feature
sk_results <- sk_esd(data_wide_t, alpha = 0.05)

cat("Scott-Knott ESD Results:\n")
print(sk_results)

# Add feature names back to Scott-Knott results
feature_names <- colnames(data_wide_t)
sk_results_df <- data.frame(Feature = feature_names, Group = sk_results$groups)

# Sort and display final results
final_results <- sk_results_df %>%
  left_join(average_feature_importance, by = "Feature") %>%
  arrange(desc(Importance))

# Replace dots with hyphens in feature names
final_results$Feature <- gsub("\\.", "-", final_results$Feature)

cat("Final Results:\n")
print(final_results)



