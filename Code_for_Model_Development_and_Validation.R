This repository provides the supplementary R code template for model development, validation, interpretation, and deployment. Predictor names have been anonymized using generic placeholders, and patient-level data are not included.

# Pipeline:
# descriptive statistics -> 8:2 stratified split -> zero-variance screening
# -> Spearman correlation screening -> Boruta + RFE -> intersection
# -> 10 candidate algorithms with stratified 5-fold CV + tuning
# -> model selection by AUC (primary), PR-AUC and Brier score (secondary)
# -> final LightGBM -> OOF Youden cutoff -> internal/external validation
# -> ROC, classification metrics, DCA, CIC, calibration, Brier
# -> SHAP -> LightGBM feature-importance weights -> model export
#
# IMPORTANT:
# - Edit the CONFIGURATION section before running.
# - Validation cohorts are not used for feature selection, tuning, model
#   selection, or cutoff determination.
# - LightGBM does not have one regression-like coefficient per predictor.
#   This script reports normalized Gain as feature-importance "weights" and
#   also exports the complete tree structure and locked model object.
# ======================================================================

# ======================================================================
# 0. PACKAGES
# ======================================================================

required_packages <- c(
  "readr", "janitor", "dplyr", "tidyr", "purrr", "tibble", "ggplot2",
  "rsample", "recipes", "parsnip", "workflows", "tune", "dials",
  "yardstick", "tidymodels", "bonsai", "discrim",
  "Boruta", "caret", "randomForest", "pROC", "lightgbm",
  "xgboost", "ranger", "rpart", "kernlab", "nnet", "naivebayes",
  "kknn", "glmnet", "MASS", "ggbeeswarm", "patchwork", "shiny"
)

missing_packages <- setdiff(required_packages, rownames(installed.packages()))
if (length(missing_packages) > 0) {
  install.packages(missing_packages, dependencies = TRUE)
}

suppressPackageStartupMessages({
  library(tidymodels)
  library(bonsai)
  library(discrim)
  library(Boruta)
  library(caret)
  library(randomForest)
  library(pROC)
  library(lightgbm)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(ggplot2)
})

tidymodels_prefer()

# ======================================================================
# 1. CONFIGURATION -- EDIT THIS SECTION ONLY
# ======================================================================

SEED <- 20260923L
INTERNAL_FILE <- "internal.csv"
EXTERNAL_FILE <- "external.csv"

# Required raw outcome coding in both input files: 0 = no event, 1 = event.
OUTCOME <- "outcome"

# Replace with the candidate predictor names used in the study.
CANDIDATE_VARS <- c(
  "predictor_01", "predictor_02", "predictor_03", "predictor_04",
  "predictor_05", "predictor_06", "predictor_07", "predictor_08",
  "predictor_09", "predictor_10", "predictor_11", "predictor_12",
  "predictor_13", "predictor_14", "predictor_15", "predictor_16",
  "predictor_17", "predictor_18", "predictor_19", "predictor_20",
  "predictor_21", "predictor_22", "predictor_23", "predictor_24"
)

# Binary/categorical predictors. Use character(0) if none.
BINARY_VARS <- c("predictor_01", "predictor_02")

# After inspecting |rho| > threshold pairs, specify variables to remove
# according to clinical meaning and completeness.
CORRELATION_THRESHOLD <- 0.80
CORRELATION_DROP <- character(0)

# Optional check; use NA_integer_ if not required.
EXPECTED_FINAL_FEATURES <- NA_integer_

TRAIN_PROP <- 0.80
CV_FOLDS <- 5L
TUNING_GRID_SIZE <- 30L
BOOTSTRAP_B <- 2000L

OUTPUT_DIR <- "reviewer_outputs"
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
out_file <- function(x) file.path(OUTPUT_DIR, x)

set.seed(SEED)

# ======================================================================
# 2. IMPORT DATA AND VALIDATE SCHEMA
# ======================================================================

internal_raw <- readr::read_csv(INTERNAL_FILE, show_col_types = FALSE) |>
  janitor::clean_names()
external_raw <- readr::read_csv(EXTERNAL_FILE, show_col_types = FALSE) |>
  janitor::clean_names()

OUTCOME <- janitor::make_clean_names(OUTCOME)
CANDIDATE_VARS <- janitor::make_clean_names(CANDIDATE_VARS)
BINARY_VARS <- janitor::make_clean_names(BINARY_VARS)
CORRELATION_DROP <- janitor::make_clean_names(CORRELATION_DROP)

required_cols <- c(OUTCOME, CANDIDATE_VARS)
stopifnot(all(required_cols %in% names(internal_raw)))
stopifnot(all(required_cols %in% names(external_raw)))
stopifnot(all(BINARY_VARS %in% CANDIDATE_VARS))
stopifnot(all(CORRELATION_DROP %in% CANDIDATE_VARS))

# Complete-case analysis as specified in the manuscript.
stopifnot(sum(is.na(internal_raw[, required_cols])) == 0)
stopifnot(sum(is.na(external_raw[, required_cols])) == 0)

stopifnot(all(internal_raw[[OUTCOME]] %in% c(0, 1)))
stopifnot(all(external_raw[[OUTCOME]] %in% c(0, 1)))

internal <- internal_raw |> select(all_of(required_cols))
external <- external_raw |> select(all_of(required_cols))

if (length(BINARY_VARS) > 0) {
  internal <- internal |> mutate(across(all_of(BINARY_VARS), as.factor))
  external <- external |> mutate(across(all_of(BINARY_VARS), as.factor))
}

# First factor level is the event of interest for yardstick.
internal[[OUTCOME]] <- factor(
  ifelse(internal[[OUTCOME]] == 1, "Event", "NoEvent"),
  levels = c("Event", "NoEvent")
)
external[[OUTCOME]] <- factor(
  ifelse(external[[OUTCOME]] == 1, "Event", "NoEvent"),
  levels = c("Event", "NoEvent")
)

cat("Internal cohort N =", nrow(internal), "\n")
cat("External cohort N =", nrow(external), "\n")
print(table(internal[[OUTCOME]]))
print(table(external[[OUTCOME]]))


# ======================================================================
# 3. BASELINE CHARACTERISTICS
# ======================================================================

continuous_vars <- setdiff(CANDIDATE_VARS, BINARY_VARS)

safe_shapiro <- function(x) {
  x <- x[is.finite(x)]
  if (length(x) < 3) return(NA_real_)
  if (length(x) > 5000) x <- x[seq_len(5000)]
  stats::shapiro.test(x)$p.value
}

compare_continuous <- function(v) {
  x <- internal[[v]]
  y <- external[[v]]
  sw_i <- safe_shapiro(x)
  sw_e <- safe_shapiro(y)
  normal <- isTRUE(sw_i >= 0.05) && isTRUE(sw_e >= 0.05)

  if (normal) {
    tt <- stats::t.test(x, y)
    tibble(
      variable = v,
      distribution = "Normal",
      internal = sprintf("%.3f ± %.3f", mean(x), sd(x)),
      external = sprintf("%.3f ± %.3f", mean(y), sd(y)),
      test = "Independent-samples t test",
      p_value = tt$p.value
    )
  } else {
    wt <- stats::wilcox.test(x, y, exact = FALSE)
    tibble(
      variable = v,
      distribution = "Non-normal",
      internal = sprintf(
        "%.3f (%.3f–%.3f)",
        median(x), quantile(x, .25), quantile(x, .75)
      ),
      external = sprintf(
        "%.3f (%.3f–%.3f)",
        median(y), quantile(y, .25), quantile(y, .75)
      ),
      test = "Mann–Whitney U test",
      p_value = wt$p.value
    )
  }
}

compare_categorical <- function(v) {
  tab <- table(
    cohort = c(rep("Internal", nrow(internal)), rep("External", nrow(external))),
    value = c(as.character(internal[[v]]), as.character(external[[v]]))
  )
  chi <- suppressWarnings(stats::chisq.test(tab, correct = FALSE))
  if (any(chi$expected < 5)) {
    p <- stats::fisher.test(tab)$p.value
    test_name <- "Fisher exact test"
  } else {
    p <- chi$p.value
    test_name <- "Chi-square test"
  }
  tibble(variable = v, test = test_name, p_value = p)
}

baseline_continuous <- map_dfr(continuous_vars, compare_continuous)
baseline_categorical <- map_dfr(BINARY_VARS, compare_categorical)

readr::write_csv(baseline_continuous, out_file("baseline_continuous.csv"))
readr::write_csv(baseline_categorical, out_file("baseline_categorical.csv"))

# ======================================================================
# 4. 8:2 STRATIFIED SPLIT OF INTERNAL COHORT
# ======================================================================

set.seed(SEED)
internal_split <- rsample::initial_split(
  internal,
  prop = TRAIN_PROP,
  strata = all_of(OUTCOME)
)

train_data <- rsample::training(internal_split)
internal_valid <- rsample::testing(internal_split)
external_valid <- external

split_summary <- bind_rows(
  tibble(
    dataset = "Training",
    n = nrow(train_data),
    events = sum(train_data[[OUTCOME]] == "Event"),
    event_rate = mean(train_data[[OUTCOME]] == "Event")
  ),
  tibble(
    dataset = "Internal validation",
    n = nrow(internal_valid),
    events = sum(internal_valid[[OUTCOME]] == "Event"),
    event_rate = mean(internal_valid[[OUTCOME]] == "Event")
  ),
  tibble(
    dataset = "External validation",
    n = nrow(external_valid),
    events = sum(external_valid[[OUTCOME]] == "Event"),
    event_rate = mean(external_valid[[OUTCOME]] == "Event")
  )
)

print(split_summary)
readr::write_csv(split_summary, out_file("cohort_split_summary.csv"))

# ======================================================================
# 5. ZERO-VARIANCE + SPEARMAN CORRELATION SCREENING
#    TRAINING DATA ONLY
# ======================================================================

zero_variance <- CANDIDATE_VARS[
  vapply(
    train_data[CANDIDATE_VARS],
    function(x) dplyr::n_distinct(x, na.rm = TRUE) <= 1L,
    logical(1)
  )
]
vars_after_zv <- setdiff(CANDIDATE_VARS, zero_variance)

readr::write_csv(
  tibble(feature = zero_variance),
  out_file("zero_variance_removed.csv")
)

# Factors are converted to integer codes only for Spearman screening.
cor_data <- train_data |>
  select(all_of(vars_after_zv)) |>
  mutate(across(where(is.factor), ~ as.numeric(.x) - 1))

if (!all(vapply(cor_data, is.numeric, logical(1)))) {
  stop("Spearman screening template requires continuous or binary predictors.")
}

cor_matrix <- stats::cor(
  cor_data,
  method = "spearman",
  use = "pairwise.complete.obs"
)

high_idx <- which(
  abs(cor_matrix) > CORRELATION_THRESHOLD & upper.tri(cor_matrix),
  arr.ind = TRUE
)

if (nrow(high_idx) > 0) {
  high_cor_pairs <- tibble(
    variable_1 = rownames(cor_matrix)[high_idx[, 1]],
    variable_2 = colnames(cor_matrix)[high_idx[, 2]],
    rho = cor_matrix[high_idx]
  ) |>
    arrange(desc(abs(rho)))
} else {
  high_cor_pairs <- tibble(
    variable_1 = character(),
    variable_2 = character(),
    rho = numeric()
  )
}

print(high_cor_pairs)
readr::write_csv(high_cor_pairs, out_file("high_correlation_pairs.csv"))

# CORRELATION_DROP is pre-specified after reviewing the high-correlation pairs
# using clinical relevance and completeness, not outcome significance.
vars_after_corr <- setdiff(vars_after_zv, CORRELATION_DROP)
stopifnot(length(vars_after_corr) >= 2)

cor_long <- as.data.frame(as.table(cor_matrix))
p_cor <- ggplot(cor_long, aes(Var1, Var2, fill = Freq)) +
  geom_tile() +
  labs(x = NULL, y = NULL, fill = "Spearman rho") +
  theme_minimal(base_size = 10) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(
  out_file("Spearman_correlation_heatmap.tiff"),
  p_cor,
  width = 9, height = 8, dpi = 600, compression = "lzw"
)

# ======================================================================
# 6. BORUTA FEATURE SELECTION
# ======================================================================

set.seed(SEED + 1L)
boruta_data <- train_data |> select(all_of(c(vars_after_corr, OUTCOME)))
boruta_formula <- stats::as.formula(paste(OUTCOME, "~ ."))

boruta_fit <- Boruta::Boruta(
  boruta_formula,
  data = boruta_data,
  maxRuns = 100,
  doTrace = 0
)

boruta_final <- Boruta::TentativeRoughFix(boruta_fit)
boruta_selected <- Boruta::getSelectedAttributes(
  boruta_final,
  withTentative = FALSE
)

readr::write_csv(
  tibble(feature = boruta_selected),
  out_file("Boruta_selected_features.csv")
)

# ======================================================================
# 7. RECURSIVE FEATURE ELIMINATION (RFE)
# ======================================================================

set.seed(SEED + 2L)

rfe_control <- caret::rfeControl(
  functions = caret::rfFuncs,
  method = "cv",
  number = CV_FOLDS,
  returnResamp = "final",
  verbose = FALSE
)

rfe_fit <- caret::rfe(
  x = train_data[, vars_after_corr, drop = FALSE],
  y = train_data[[OUTCOME]],
  sizes = seq_along(vars_after_corr),
  metric = "Accuracy",
  rfeControl = rfe_control
)

rfe_selected <- caret::predictors(rfe_fit)

readr::write_csv(
  tibble(feature = rfe_selected),
  out_file("RFE_selected_features.csv")
)

readr::write_csv(
  as.data.frame(rfe_fit$results),
  out_file("RFE_performance_by_subset_size.csv")
)

# ======================================================================
# 8. BORUTA ∩ RFE
# ======================================================================

final_features <- intersect(boruta_selected, rfe_selected)

if (!is.na(EXPECTED_FINAL_FEATURES)) {
  stopifnot(length(final_features) == EXPECTED_FINAL_FEATURES)
}
if (length(final_features) < 1) {
  stop("Boruta and RFE have no common selected features.")
}

cat("Boruta selected:", length(boruta_selected), "\n")
cat("RFE selected:", length(rfe_selected), "\n")
cat("Intersection:", length(final_features), "\n")
print(final_features)

readr::write_csv(
  tibble(feature = final_features),
  out_file("final_selected_features.csv")
)


# ======================================================================
# 9. FINAL MODELING DATA + PREPROCESSING RECIPE
# ======================================================================

model_cols <- c(OUTCOME, final_features)

train_model <- train_data |> select(all_of(model_cols))
internal_model <- internal_valid |> select(all_of(model_cols))
external_model <- external_valid |> select(all_of(model_cols))

model_formula <- stats::as.formula(paste(OUTCOME, "~ ."))

# Preprocessing is estimated within each resample by the workflow.
# This reproduces zero-variance protection + dummy/one-hot encoding.
model_recipe <- recipes::recipe(model_formula, data = train_model) |>
  recipes::step_zv(recipes::all_predictors()) |>
  recipes::step_dummy(
    recipes::all_nominal_predictors(),
    one_hot = TRUE
  )

# ======================================================================
# 10. STRATIFIED 5-FOLD CV + MODEL-SELECTION METRICS
# ======================================================================

set.seed(SEED + 3L)
folds5 <- rsample::vfold_cv(
  train_model,
  v = CV_FOLDS,
  strata = all_of(OUTCOME)
)

model_metrics <- yardstick::metric_set(
  yardstick::roc_auc,
  yardstick::pr_auc,
  yardstick::brier_class
)

# ======================================================================
# 11. TEN CANDIDATE ALGORITHMS
# ======================================================================

spec_xgboost <- parsnip::boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  mtry = tune(),
  min_n = tune(),
  loss_reduction = tune(),
  sample_size = tune()
) |>
  parsnip::set_engine("xgboost") |>
  parsnip::set_mode("classification")

spec_svm <- parsnip::svm_rbf(
  cost = tune(),
  rbf_sigma = tune()
) |>
  parsnip::set_engine("kernlab") |>
  parsnip::set_mode("classification")

spec_rpart <- parsnip::decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
) |>
  parsnip::set_engine("rpart") |>
  parsnip::set_mode("classification")

spec_ranger <- parsnip::rand_forest(
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) |>
  parsnip::set_engine("ranger", probability = TRUE) |>
  parsnip::set_mode("classification")

spec_nnet <- parsnip::mlp(
  hidden_units = tune(),
  penalty = tune(),
  epochs = tune()
) |>
  parsnip::set_engine("nnet", trace = FALSE, MaxNWts = 10000) |>
  parsnip::set_mode("classification")

spec_nb <- discrim::naive_Bayes(
  smoothness = tune(),
  Laplace = tune()
) |>
  parsnip::set_engine("naivebayes") |>
  parsnip::set_mode("classification")

spec_lightgbm <- parsnip::boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  mtry = tune(),
  min_n = tune(),
  loss_reduction = tune()
) |>
  parsnip::set_engine(
    "lightgbm",
    deterministic = TRUE,
    num_threads = 1L
  ) |>
  parsnip::set_mode("classification")

spec_lda <- discrim::discrim_linear() |>
  parsnip::set_engine("MASS") |>
  parsnip::set_mode("classification")

spec_kknn <- parsnip::nearest_neighbor(
  neighbors = tune(),
  weight_func = tune(),
  dist_power = tune()
) |>
  parsnip::set_engine("kknn") |>
  parsnip::set_mode("classification")

spec_glmnet <- parsnip::logistic_reg(
  penalty = tune(),
  mixture = tune()
) |>
  parsnip::set_engine("glmnet") |>
  parsnip::set_mode("classification")

make_workflow <- function(spec) {
  workflows::workflow() |>
    workflows::add_recipe(model_recipe) |>
    workflows::add_model(spec)
}

workflow_list <- list(
  xgboost = make_workflow(spec_xgboost),
  svm_rbf = make_workflow(spec_svm),
  cart = make_workflow(spec_rpart),
  ranger = make_workflow(spec_ranger),
  neural_network = make_workflow(spec_nnet),
  naive_bayes = make_workflow(spec_nb),
  lightgbm = make_workflow(spec_lightgbm),
  lda = make_workflow(spec_lda),
  knn = make_workflow(spec_kknn),
  elastic_net_logistic = make_workflow(spec_glmnet)
)

# ======================================================================
# 12. HYPERPARAMETER TUNING
# ======================================================================

ctrl_grid <- tune::control_grid(
  save_pred = TRUE,
  save_workflow = TRUE,
  verbose = TRUE
)

ctrl_resamples <- tune::control_resamples(
  save_pred = TRUE,
  save_workflow = TRUE,
  verbose = TRUE
)

predictor_data_for_finalization <- train_model |>
  select(-all_of(OUTCOME))

model_results <- list()

for (i in seq_along(workflow_list)) {
  model_name <- names(workflow_list)[i]
  wf <- workflow_list[[i]]

  cat("\nRunning:", model_name, "\n")
  set.seed(SEED + 100L + i)

  if (model_name == "lda") {
    model_results[[model_name]] <- tune::fit_resamples(
      wf,
      resamples = folds5,
      metrics = model_metrics,
      control = ctrl_resamples
    )
  } else {
    param_set <- workflows::extract_parameter_set_dials(wf)
    param_set <- dials::finalize(
      param_set,
      predictor_data_for_finalization
    )

    model_results[[model_name]] <- tune::tune_grid(
      wf,
      resamples = folds5,
      param_info = param_set,
      grid = TUNING_GRID_SIZE,
      metrics = model_metrics,
      control = ctrl_grid
    )
  }
}

# ======================================================================
# 13. EXTRACT BEST RESULT FROM EACH ALGORITHM
# ======================================================================

extract_best_performance <- function(result, model_name) {
  if (model_name == "lda") {
    metrics <- tune::collect_metrics(result)
    auc <- metrics |> filter(.metric == "roc_auc") |> pull(mean)
    pr <- metrics |> filter(.metric == "pr_auc") |> pull(mean)
    br <- metrics |> filter(.metric == "brier_class") |> pull(mean)
    return(tibble(Model = model_name, AUC = auc, PR_AUC = pr, Brier = br))
  }

  best <- tune::select_best(result, metric = "roc_auc")
  param_names <- setdiff(names(best), ".config")

  metrics <- tune::collect_metrics(result) |>
    semi_join(best, by = param_names)

  tibble(
    Model = model_name,
    AUC = metrics |> filter(.metric == "roc_auc") |> pull(mean) |> first(),
    PR_AUC = metrics |> filter(.metric == "pr_auc") |> pull(mean) |> first(),
    Brier = metrics |> filter(.metric == "brier_class") |> pull(mean) |> first()
  )
}

best_results <- purrr::imap_dfr(
  model_results,
  ~ extract_best_performance(.x, .y)
)

# AUC is the primary criterion; PR-AUC and Brier are secondary.
best_results_ranked <- best_results |>
  arrange(desc(AUC), desc(PR_AUC), Brier)

print(best_results_ranked)
readr::write_csv(
  best_results_ranked,
  out_file("candidate_model_comparison.csv")
)

final_algorithm <- best_results_ranked$Model[1]
cat("Selected algorithm:", final_algorithm, "\n")

if (final_algorithm != "lightgbm") {
  warning(
    "The highest cross-validated AUC is not from LightGBM. ",
    "Do not force LightGBM unless this matches the actual analysis."
  )
}

# ======================================================================
# 14. LOCK THE FINAL LIGHTGBM HYPERPARAMETERS
# ======================================================================

best_lgbm_params <- tune::select_best(
  model_results$lightgbm,
  metric = "roc_auc"
)

final_lgbm_workflow <- tune::finalize_workflow(
  workflow_list$lightgbm,
  best_lgbm_params
)

readr::write_csv(
  best_lgbm_params,
  out_file("LightGBM_locked_hyperparameters.csv")
)

# ======================================================================
# 15. OOF PREDICTIONS FROM THE LOCKED LIGHTGBM + YOUDEN CUTOFF
# ======================================================================

set.seed(SEED + 500L)

lgbm_oof_res <- tune::fit_resamples(
  final_lgbm_workflow,
  resamples = folds5,
  metrics = model_metrics,
  control = tune::control_resamples(save_pred = TRUE)
)

oof_predictions <- tune::collect_predictions(
  lgbm_oof_res,
  summarize = TRUE
)

stopifnot(".pred_Event" %in% names(oof_predictions))

oof_y <- as.integer(oof_predictions[[OUTCOME]] == "Event")
oof_prob <- oof_predictions$.pred_Event

roc_oof <- pROC::roc(
  response = oof_y,
  predictor = oof_prob,
  levels = c(0, 1),
  direction = "<",
  quiet = TRUE
)

youden <- pROC::coords(
  roc_oof,
  x = "best",
  best.method = "youden",
  ret = c("threshold", "sensitivity", "specificity"),
  transpose = FALSE
)

cutoff <- as.numeric(youden$threshold[1])
cat("Locked cutoff =", cutoff, "\n")

readr::write_csv(
  tibble(
    cutoff = cutoff,
    sensitivity = as.numeric(youden$sensitivity[1]),
    specificity = as.numeric(youden$specificity[1])
  ),
  out_file("OOF_Youden_cutoff.csv")
)

# ======================================================================
# 16. REFIT AND LOCK FINAL LIGHTGBM ON THE COMPLETE TRAINING SET
# ======================================================================

set.seed(SEED + 600L)
final_fit <- workflows::fit(
  final_lgbm_workflow,
  data = train_model
)

# Use the workflow for prediction so the fitted preprocessing recipe is retained.
prob_internal <- predict(
  final_fit,
  new_data = internal_model,
  type = "prob"
)$.pred_Event

prob_external <- predict(
  final_fit,
  new_data = external_model,
  type = "prob"
)$.pred_Event

y_internal <- as.integer(internal_model[[OUTCOME]] == "Event")
y_external <- as.integer(external_model[[OUTCOME]] == "Event")

stopifnot(all(prob_internal >= 0 & prob_internal <= 1))
stopifnot(all(prob_external >= 0 & prob_external <= 1))

readr::write_csv(
  tibble(outcome = y_internal, predicted_probability = prob_internal),
  out_file("internal_validation_predictions.csv")
)
readr::write_csv(
  tibble(outcome = y_external, predicted_probability = prob_external),
  out_file("external_validation_predictions.csv")
)


# ======================================================================
# 17. VALIDATION METRICS: ROC/AUC + 95% CI
# ======================================================================

roc_with_ci <- function(y, prob) {
  roc_obj <- pROC::roc(
    response = y,
    predictor = prob,
    levels = c(0, 1),
    direction = "<",
    quiet = TRUE
  )
  ci <- pROC::ci.auc(roc_obj, method = "delong")
  list(
    roc = roc_obj,
    auc = as.numeric(pROC::auc(roc_obj)),
    lower = as.numeric(ci[1]),
    upper = as.numeric(ci[3])
  )
}

roc_internal <- roc_with_ci(y_internal, prob_internal)
roc_external <- roc_with_ci(y_external, prob_external)

auc_table <- tibble(
  dataset = c("Internal validation", "External validation"),
  AUC = c(roc_internal$auc, roc_external$auc),
  Lower95CI = c(roc_internal$lower, roc_external$lower),
  Upper95CI = c(roc_internal$upper, roc_external$upper)
)

print(auc_table)
readr::write_csv(auc_table, out_file("ROC_AUC_results.csv"))

plot_roc <- function(roc_result, title_text) {
  ggplot(
    tibble(
      FPR = 1 - roc_result$roc$specificities,
      TPR = roc_result$roc$sensitivities
    ),
    aes(FPR, TPR)
  ) +
    geom_line(linewidth = 1) +
    geom_abline(intercept = 0, slope = 1, linetype = 2) +
    annotate(
      "text",
      x = 0.60, y = 0.20,
      label = sprintf(
        "AUC = %.3f (95%% CI %.3f–%.3f)",
        roc_result$auc, roc_result$lower, roc_result$upper
      )
    ) +
    coord_equal() +
    labs(title = title_text, x = "1 - Specificity", y = "Sensitivity") +
    theme_classic(base_size = 12)
}

ggsave(
  out_file("ROC_internal_validation.tiff"),
  plot_roc(roc_internal, "Internal validation"),
  width = 6, height = 6, dpi = 600, compression = "lzw"
)

ggsave(
  out_file("ROC_external_validation.tiff"),
  plot_roc(roc_external, "External validation"),
  width = 6, height = 6, dpi = 600, compression = "lzw"
)

# ======================================================================
# 18. CONFUSION MATRIX AND PRE-SPECIFIED-CUTOFF METRICS
# ======================================================================

classification_metrics <- function(y, prob, cutoff) {
  pred <- ifelse(prob >= cutoff, 1L, 0L)

  TP <- sum(pred == 1 & y == 1)
  TN <- sum(pred == 0 & y == 0)
  FP <- sum(pred == 1 & y == 0)
  FN <- sum(pred == 0 & y == 1)

  sensitivity <- ifelse(TP + FN > 0, TP / (TP + FN), NA_real_)
  specificity <- ifelse(TN + FP > 0, TN / (TN + FP), NA_real_)
  precision <- ifelse(TP + FP > 0, TP / (TP + FP), NA_real_)
  accuracy <- (TP + TN) / length(y)
  f1 <- ifelse(
    is.finite(precision + sensitivity) && (precision + sensitivity) > 0,
    2 * precision * sensitivity / (precision + sensitivity),
    NA_real_
  )

  tibble(
    TP = TP, TN = TN, FP = FP, FN = FN,
    Sensitivity = sensitivity,
    Specificity = specificity,
    Precision = precision,
    F1_score = f1,
    Accuracy = accuracy
  )
}

metrics_internal <- classification_metrics(y_internal, prob_internal, cutoff)
metrics_external <- classification_metrics(y_external, prob_external, cutoff)

readr::write_csv(
  metrics_internal,
  out_file("classification_metrics_internal.csv")
)
readr::write_csv(
  metrics_external,
  out_file("classification_metrics_external.csv")
)

print(metrics_internal)
print(metrics_external)

# ======================================================================
# 19. DECISION CURVE ANALYSIS (DCA)
# ======================================================================

calculate_dca <- function(y, prob, thresholds = seq(0.01, 0.99, 0.01)) {
  n <- length(y)
  prevalence <- mean(y)

  map_dfr(thresholds, function(pt) {
    pred <- prob >= pt
    TP <- sum(pred & y == 1)
    FP <- sum(pred & y == 0)

    nb_model <- TP / n - FP / n * pt / (1 - pt)
    nb_all <- prevalence - (1 - prevalence) * pt / (1 - pt)

    tibble(
      threshold = pt,
      Model = nb_model,
      Treat_all = nb_all,
      Treat_none = 0
    )
  })
}

plot_dca <- function(dca_data, title_text) {
  dca_data |>
    pivot_longer(
      cols = c(Model, Treat_all, Treat_none),
      names_to = "Strategy",
      values_to = "NetBenefit"
    ) |>
    ggplot(aes(threshold, NetBenefit, linetype = Strategy)) +
    geom_line(linewidth = 0.9) +
    labs(
      title = title_text,
      x = "Threshold probability",
      y = "Net benefit",
      linetype = NULL
    ) +
    theme_classic(base_size = 12)
}

dca_internal <- calculate_dca(y_internal, prob_internal)
dca_external <- calculate_dca(y_external, prob_external)

ggsave(
  out_file("DCA_internal_validation.tiff"),
  plot_dca(dca_internal, "Internal validation"),
  width = 6, height = 5, dpi = 600, compression = "lzw"
)
ggsave(
  out_file("DCA_external_validation.tiff"),
  plot_dca(dca_external, "External validation"),
  width = 6, height = 5, dpi = 600, compression = "lzw"
)

# ======================================================================
# 20. CLINICAL IMPACT CURVE (CIC)
# ======================================================================

calculate_cic <- function(y, prob, thresholds = seq(0.01, 0.99, 0.01)) {
  n <- length(y)

  map_dfr(thresholds, function(pt) {
    high <- prob >= pt
    tibble(
      threshold = pt,
      high_risk_per_1000 = sum(high) / n * 1000,
      event_among_high_risk_per_1000 = sum(high & y == 1) / n * 1000
    )
  })
}

plot_cic <- function(cic_data, title_text) {
  cic_data |>
    pivot_longer(
      cols = c(high_risk_per_1000, event_among_high_risk_per_1000),
      names_to = "Series",
      values_to = "Count"
    ) |>
    ggplot(aes(threshold, Count, linetype = Series)) +
    geom_line(linewidth = 0.9) +
    labs(
      title = title_text,
      x = "Threshold probability",
      y = "Number per 1000",
      linetype = NULL
    ) +
    theme_classic(base_size = 12)
}

cic_internal <- calculate_cic(y_internal, prob_internal)
cic_external <- calculate_cic(y_external, prob_external)

ggsave(
  out_file("CIC_internal_validation.tiff"),
  plot_cic(cic_internal, "Internal validation"),
  width = 6, height = 5, dpi = 600, compression = "lzw"
)
ggsave(
  out_file("CIC_external_validation.tiff"),
  plot_cic(cic_external, "External validation"),
  width = 6, height = 5, dpi = 600, compression = "lzw"
)

# ======================================================================
# 21. CALIBRATION INTERCEPT, SLOPE, C-STATISTIC
# ======================================================================

calibration_statistics <- function(y, prob) {
  eps <- 1e-6
  p <- pmin(pmax(prob, eps), 1 - eps)
  lp <- qlogis(p)

  cal_model <- glm(y ~ lp, family = binomial())
  co <- coef(cal_model)
  se <- sqrt(diag(vcov(cal_model)))

  roc_obj <- pROC::roc(
    y, prob,
    levels = c(0, 1),
    direction = "<",
    quiet = TRUE
  )
  auc_ci <- as.numeric(pROC::ci.auc(roc_obj, method = "delong"))

  tibble(
    Metric = c("Calibration intercept", "Calibration slope", "C-statistic"),
    Estimate = c(co[1], co[2], as.numeric(pROC::auc(roc_obj))),
    Lower95CI = c(
      co[1] - 1.96 * se[1],
      co[2] - 1.96 * se[2],
      auc_ci[1]
    ),
    Upper95CI = c(
      co[1] + 1.96 * se[1],
      co[2] + 1.96 * se[2],
      auc_ci[3]
    )
  )
}

cal_internal <- calibration_statistics(y_internal, prob_internal)
cal_external <- calibration_statistics(y_external, prob_external)

readr::write_csv(cal_internal, out_file("calibration_statistics_internal.csv"))
readr::write_csv(cal_external, out_file("calibration_statistics_external.csv"))

make_calibration_plot <- function(y, prob, stat_table, title_text) {
  dat <- tibble(outcome = y, prob = prob)

  intercept <- stat_table |> filter(Metric == "Calibration intercept")
  slope <- stat_table |> filter(Metric == "Calibration slope")
  cstat <- stat_table |> filter(Metric == "C-statistic")

  label <- paste0(
    "Intercept: ", sprintf("%.2f", intercept$Estimate),
    " (", sprintf("%.2f", intercept$Lower95CI), " to ",
    sprintf("%.2f", intercept$Upper95CI), ")\n",
    "Slope: ", sprintf("%.2f", slope$Estimate),
    " (", sprintf("%.2f", slope$Lower95CI), " to ",
    sprintf("%.2f", slope$Upper95CI), ")\n",
    "C-statistic: ", sprintf("%.2f", cstat$Estimate),
    " (", sprintf("%.2f", cstat$Lower95CI), " to ",
    sprintf("%.2f", cstat$Upper95CI), ")"
  )

  ggplot(dat, aes(prob, outcome)) +
    geom_abline(intercept = 0, slope = 1, linetype = 2) +
    geom_smooth(
      method = "loess", formula = y ~ x,
      se = TRUE, level = 0.95, span = 0.75
    ) +
    annotate("text", x = 0.03, y = 0.97, label = label, hjust = 0, vjust = 1) +
    scale_x_continuous(limits = c(0, 1)) +
    coord_cartesian(ylim = c(0, 1)) +
    labs(
      title = title_text,
      x = "Predicted probability",
      y = "Observed proportion"
    ) +
    theme_classic(base_size = 12)
}

ggsave(
  out_file("Calibration_internal_validation.tiff"),
  make_calibration_plot(y_internal, prob_internal, cal_internal, "Internal validation"),
  width = 6, height = 6, dpi = 600, compression = "lzw"
)

ggsave(
  out_file("Calibration_external_validation.tiff"),
  make_calibration_plot(y_external, prob_external, cal_external, "External validation"),
  width = 6, height = 6, dpi = 600, compression = "lzw"
)

# ======================================================================
# 22. BRIER SCORE + BOOTSTRAP 95% CI
# ======================================================================

brier_ci <- function(y, prob, B = BOOTSTRAP_B, seed = SEED) {
  estimate <- mean((prob - y)^2)

  set.seed(seed)
  n <- length(y)
  boot_values <- replicate(B, {
    idx <- sample.int(n, size = n, replace = TRUE)
    mean((prob[idx] - y[idx])^2)
  })

  ci <- quantile(boot_values, c(0.025, 0.975), na.rm = TRUE)

  tibble(
    Brier = estimate,
    Lower95CI = unname(ci[1]),
    Upper95CI = unname(ci[2])
  )
}

brier_internal <- brier_ci(
  y_internal, prob_internal,
  B = BOOTSTRAP_B, seed = SEED + 700L
)
brier_external <- brier_ci(
  y_external, prob_external,
  B = BOOTSTRAP_B, seed = SEED + 701L
)

readr::write_csv(brier_internal, out_file("Brier_internal.csv"))
readr::write_csv(brier_external, out_file("Brier_external.csv"))

# Optional binned visualization used only for display; the Brier score itself
# is computed from individual-level probabilities above.
make_brier_bins <- function(y, prob, breaks = seq(0, 1, by = 0.2)) {
  tibble(outcome = y, probability = prob) |>
    mutate(
      bin = cut(
        probability,
        breaks = breaks,
        include.lowest = TRUE,
        labels = FALSE
      )
    ) |>
    group_by(bin) |>
    summarise(
      N = n(),
      Mean_predicted = mean(probability),
      Event_rate = mean(outcome),
      .groups = "drop"
    ) |>
    mutate(
      Bin_midpoint = (breaks[bin] + breaks[bin + 1]) / 2
    )
}

plot_brier_bins <- function(y, prob, brier_table, title_text) {
  bins <- make_brier_bins(y, prob)
  label <- sprintf(
    "Brier score = %.3f (95%% CI %.3f–%.3f)",
    brier_table$Brier,
    brier_table$Lower95CI,
    brier_table$Upper95CI
  )

  ggplot(bins, aes(Bin_midpoint, Event_rate)) +
    geom_abline(intercept = 0, slope = 1, linetype = 2) +
    geom_line(aes(group = 1), linewidth = 0.8) +
    geom_point(size = 2.5) +
    annotate("text", x = 0.98, y = 0.05, label = label, hjust = 1) +
    scale_x_continuous(limits = c(0, 1)) +
    scale_y_continuous(limits = c(0, 1)) +
    labs(title = title_text, x = "Bin midpoint", y = "Event rate") +
    theme_classic(base_size = 12)
}

ggsave(
  out_file("Brier_plot_internal_validation.tiff"),
  plot_brier_bins(y_internal, prob_internal, brier_internal, "Internal validation"),
  width = 6, height = 6, dpi = 600, compression = "lzw"
)

ggsave(
  out_file("Brier_plot_external_validation.tiff"),
  plot_brier_bins(y_external, prob_external, brier_external, "External validation"),
  width = 6, height = 6, dpi = 600, compression = "lzw"
)


# ======================================================================
# 23. SHAP INTERPRETATION ON THE INTERNAL VALIDATION SET
# ======================================================================

# Extract the fitted preprocessing recipe and the fitted LightGBM engine.
# Predictions for performance evaluation should still use final_fit (workflow).
fitted_recipe <- workflows::extract_recipe(final_fit, estimated = TRUE)
final_lgbm_engine <- workflows::extract_fit_engine(final_fit)

# Bake the internal validation predictors exactly as seen by LightGBM.
X_internal_baked <- recipes::bake(
  fitted_recipe,
  new_data = internal_model,
  recipes::all_predictors(),
  composition = "matrix"
)

storage.mode(X_internal_baked) <- "double"

shap_full <- predict(
  final_lgbm_engine,
  X_internal_baked,
  type = "contrib"
)

shap_full <- as.matrix(shap_full)
n_features_baked <- ncol(X_internal_baked)

stopifnot(ncol(shap_full) == n_features_baked + 1L)

shap_matrix <- shap_full[, seq_len(n_features_baked), drop = FALSE]
shap_base <- shap_full[, n_features_baked + 1L]

colnames(shap_matrix) <- colnames(X_internal_baked)

# Verify SHAP additivity on the raw LightGBM scale.
raw_prediction <- predict(
  final_lgbm_engine,
  X_internal_baked,
  type = "raw"
)

shap_reconstruction_error <- max(
  abs(raw_prediction - (rowSums(shap_matrix) + shap_base)),
  na.rm = TRUE
)

cat("Maximum SHAP reconstruction error =", shap_reconstruction_error, "\n")

# Mean absolute SHAP importance.
shap_importance <- tibble(
  Feature = colnames(shap_matrix),
  MeanAbsSHAP = colMeans(abs(shap_matrix), na.rm = TRUE)
) |>
  arrange(desc(MeanAbsSHAP)) |>
  mutate(
    SHAP_weight = MeanAbsSHAP / sum(MeanAbsSHAP),
    SHAP_weight_percent = 100 * SHAP_weight
  )

readr::write_csv(
  shap_importance,
  out_file("SHAP_feature_importance.csv")
)

# Long-form data for SHAP summary plot.
shap_long <- as.data.frame(shap_matrix) |>
  mutate(row_id = row_number()) |>
  pivot_longer(
    cols = -row_id,
    names_to = "Feature",
    values_to = "SHAP"
  )

feature_long <- as.data.frame(X_internal_baked) |>
  mutate(row_id = row_number()) |>
  pivot_longer(
    cols = -row_id,
    names_to = "Feature",
    values_to = "FeatureValue"
  )

scale_01 <- function(x) {
  rg <- range(x, na.rm = TRUE)
  if (!all(is.finite(rg)) || diff(rg) == 0) {
    return(rep(0.5, length(x)))
  }
  (x - rg[1]) / diff(rg)
}

shap_summary_data <- shap_long |>
  left_join(feature_long, by = c("row_id", "Feature")) |>
  group_by(Feature) |>
  mutate(FeatureValue_scaled = scale_01(FeatureValue)) |>
  ungroup()

feature_order <- shap_importance$Feature
shap_summary_data$Feature <- factor(
  shap_summary_data$Feature,
  levels = rev(feature_order)
)

p_shap_bar <- shap_importance |>
  mutate(Feature = factor(Feature, levels = rev(Feature))) |>
  ggplot(aes(MeanAbsSHAP, Feature)) +
  geom_col(width = 0.7) +
  labs(x = "mean(|SHAP value|)", y = NULL) +
  theme_classic(base_size = 12)

p_shap_summary <- ggplot(
  shap_summary_data,
  aes(SHAP, Feature, color = FeatureValue_scaled)
) +
  geom_vline(xintercept = 0, linewidth = 0.4) +
  ggbeeswarm::geom_quasirandom(
    orientation = "y",
    width = 0.30,
    size = 1.5,
    alpha = 0.85
  ) +
  scale_color_gradient(
    low = "grey20",
    high = "grey80",
    limits = c(0, 1),
    breaks = c(0, 1),
    labels = c("Low", "High"),
    name = "Feature value"
  ) +
  labs(x = "SHAP value", y = NULL) +
  theme_classic(base_size = 12)

Figure_SHAP <- p_shap_bar + p_shap_summary +
  patchwork::plot_annotation(tag_levels = "A")

ggsave(
  out_file("SHAP_global_interpretation.tiff"),
  Figure_SHAP,
  width = 14, height = 6, dpi = 600, compression = "lzw"
)

readr::write_csv(
  shap_summary_data,
  out_file("SHAP_summary_internal_validation.csv")
)

# ======================================================================
# 24. LIGHTGBM FEATURE IMPORTANCE / "MODEL WEIGHTS"
# ======================================================================

# For a tree ensemble, these are feature-importance weights, not coefficients.
importance_raw <- lightgbm::lgb.importance(
  final_lgbm_engine,
  percentage = FALSE
) |>
  as.data.frame()

importance_weights <- importance_raw |>
  mutate(
    Gain_weight = ifelse(sum(Gain) > 0, Gain / sum(Gain), NA_real_),
    Gain_percent = 100 * Gain_weight,
    Frequency_weight = ifelse(
      sum(Frequency) > 0,
      Frequency / sum(Frequency),
      NA_real_
    ),
    Frequency_percent = 100 * Frequency_weight
  ) |>
  arrange(desc(Gain_weight))

print(importance_weights)

readr::write_csv(
  importance_weights,
  out_file("LightGBM_feature_importance_weights.csv")
)

# Full tree structure contains split thresholds and leaf values and is the
# closest transparent representation of the actual fitted tree ensemble.
tree_structure <- lightgbm::lgb.model.dt.tree(final_lgbm_engine) |>
  as.data.frame()

readr::write_csv(
  tree_structure,
  out_file("LightGBM_complete_tree_structure.csv")
)

# ======================================================================
# 25. SAVE LOCKED MODEL + METADATA
# ======================================================================

# Save the complete workflow (recipe + model). This is the preferred R object
# for future prediction because preprocessing is retained.
saveRDS(
  final_fit,
  out_file("locked_prediction_workflow.rds")
)

# Also save the native LightGBM engine for independent inspection.
lightgbm::lgb.save(
  final_lgbm_engine,
  filename = out_file("locked_LightGBM_model.txt")
)

metadata <- list(
  outcome = OUTCOME,
  positive_class = "Event",
  candidate_predictors = CANDIDATE_VARS,
  zero_variance_removed = zero_variance,
  correlation_threshold = CORRELATION_THRESHOLD,
  correlation_removed = CORRELATION_DROP,
  boruta_selected = boruta_selected,
  rfe_selected = rfe_selected,
  final_features = final_features,
  selected_algorithm = final_algorithm,
  locked_hyperparameters = best_lgbm_params,
  cutoff = cutoff,
  seed = SEED,
  R_version = R.version.string,
  package_versions = sapply(
    c("tidymodels", "bonsai", "lightgbm", "Boruta", "caret", "pROC"),
    function(pkg) as.character(utils::packageVersion(pkg))
  )
)

saveRDS(
  metadata,
  out_file("model_metadata.rds")
)

capture.output(
  metadata,
  file = out_file("model_metadata.txt")
)

capture.output(
  sessionInfo(),
  file = out_file("sessionInfo.txt")
)

# ======================================================================
# 26. GENERIC SINGLE-PATIENT PREDICTION FUNCTION
# ======================================================================

predict_one_patient <- function(patient, workflow_fit, cutoff) {
  stopifnot(is.data.frame(patient), nrow(patient) == 1L)

  prob <- predict(
    workflow_fit,
    new_data = patient,
    type = "prob"
  )$.pred_Event

  tibble(
    predicted_probability = as.numeric(prob),
    cutoff = cutoff,
    risk_group = ifelse(prob >= cutoff, "High risk", "Low risk")
  )
}

# Example usage (not executed):
# new_patient <- data.frame(
#   predictor_01 = ...,
#   predictor_02 = ...,
#   ...
# )
# predict_one_patient(new_patient, final_fit, cutoff)

# ======================================================================
# 27. OPTIONAL GENERIC SHINY CALCULATOR
# ======================================================================
# This function creates a dynamic calculator from the final raw predictor
# names. It is intended as a deployment template; input labels/ranges should
# be customized using clinically meaningful units before public deployment.

run_generic_shiny_calculator <- function(
  workflow_path = out_file("locked_prediction_workflow.rds"),
  metadata_path = out_file("model_metadata.rds")
) {
  library(shiny)

  wf <- readRDS(workflow_path)
  md <- readRDS(metadata_path)
  features <- md$final_features
  fixed_cutoff <- md$cutoff

  ui <- fluidPage(
    titlePanel("Clinical Risk Calculator"),
    sidebarLayout(
      sidebarPanel(
        uiOutput("feature_inputs"),
        actionButton("calculate", "Calculate")
      ),
      mainPanel(
        verbatimTextOutput("result")
      )
    )
  )

  server <- function(input, output, session) {
    output$feature_inputs <- renderUI({
      tagList(
        lapply(features, function(f) {
          numericInput(
            inputId = paste0("x__", f),
            label = f,
            value = 0
          )
        })
      )
    })

    result <- eventReactive(input$calculate, {
      vals <- lapply(features, function(f) {
        input[[paste0("x__", f)]]
      })
      names(vals) <- features

      patient <- as.data.frame(vals, check.names = FALSE)

      predict_one_patient(
        patient = patient,
        workflow_fit = wf,
        cutoff = fixed_cutoff
      )
    })

    output$result <- renderPrint({
      req(result())
      result()
    })
  }

  shinyApp(ui, server)
}

# To launch locally after the analysis:
# run_generic_shiny_calculator()

# ======================================================================
# 28. FINAL REPRODUCIBILITY CHECKLIST
# ======================================================================

cat("\nAnalysis completed. Reviewer outputs are in:", OUTPUT_DIR, "\n")
cat("Final selected features:", length(final_features), "\n")
cat("Selected algorithm:", final_algorithm, "\n")
cat("Locked cutoff:", cutoff, "\n")
cat("Model weights file: LightGBM_feature_importance_weights.csv\n")
cat("Full tree file: LightGBM_complete_tree_structure.csv\n")
cat("Locked workflow: locked_prediction_workflow.rds\n")
cat("Native LightGBM model: locked_LightGBM_model.txt\n")

# ======================================================================
# 29. PREPARE SHINY APP FOR SHINYAPPS.IO DEPLOYMENT
# ======================================================================

# This section creates a standalone Shiny application directory containing:
#   1. app.R
#   2. locked_prediction_workflow.rds
#   3. model_metadata.rds
#
# The application can then be deployed to shinyapps.io using rsconnect.

SHINY_APP_DIR <- "LRI_risk_calculator"
SHINY_APP_NAME <- "LRI-risk-calculator"

dir.create(
  SHINY_APP_DIR,
  showWarnings = FALSE,
  recursive = TRUE
)

# ----------------------------------------------------------------------
# 29.1 Copy the locked prediction model and metadata to the app directory
# ----------------------------------------------------------------------

workflow_source <- out_file("locked_prediction_workflow.rds")
metadata_source <- out_file("model_metadata.rds")

if (!file.exists(workflow_source)) {
  stop("Cannot find locked_prediction_workflow.rds")
}

if (!file.exists(metadata_source)) {
  stop("Cannot find model_metadata.rds")
}

file.copy(
  from = workflow_source,
  to = file.path(
    SHINY_APP_DIR,
    "locked_prediction_workflow.rds"
  ),
  overwrite = TRUE
)

file.copy(
  from = metadata_source,
  to = file.path(
    SHINY_APP_DIR,
    "model_metadata.rds"
  ),
  overwrite = TRUE
)


# ----------------------------------------------------------------------
# 29.2 Save default input values
# ----------------------------------------------------------------------
# Median values in the training set are used only as initial values
# displayed in the calculator. They are NOT used for model fitting.

numeric_final_features <- final_features[
  vapply(
    train_model[final_features],
    is.numeric,
    logical(1)
  )
]

input_defaults <- vapply(
  numeric_final_features,
  function(v) {
    median(train_model[[v]], na.rm = TRUE)
  },
  numeric(1)
)

saveRDS(
  input_defaults,
  file.path(
    SHINY_APP_DIR,
    "input_defaults.rds"
  )
)


# ----------------------------------------------------------------------
# 29.3 Generate app.R
# ----------------------------------------------------------------------

app_code <- '
# ======================================================================
# LRI RISK CALCULATOR
# ======================================================================

library(shiny)
library(tidymodels)
library(bonsai)
library(lightgbm)

# ----------------------------------------------------------------------
# Load locked model and metadata
# ----------------------------------------------------------------------

wf <- readRDS("locked_prediction_workflow.rds")
md <- readRDS("model_metadata.rds")
input_defaults <- readRDS("input_defaults.rds")

features <- md$final_features
fixed_cutoff <- md$cutoff


# ----------------------------------------------------------------------
# User interface
# ----------------------------------------------------------------------

ui <- fluidPage(

  tags$head(
    tags$style(HTML("
      body {
        font-family: Arial, sans-serif;
      }

      .title {
        font-size: 30px;
        font-weight: bold;
        margin-bottom: 5px;
      }

      .subtitle {
        color: #666666;
        margin-bottom: 25px;
      }

      .result-box {
        padding: 20px;
        border: 1px solid #dddddd;
        border-radius: 8px;
        margin-top: 15px;
      }

      .probability {
        font-size: 32px;
        font-weight: bold;
      }

      .risk-high {
        font-size: 24px;
        font-weight: bold;
      }

      .risk-low {
        font-size: 24px;
        font-weight: bold;
      }
    "))
  ),

  div(
    class = "title",
    "Late Recurrent Intussusception Risk Calculator"
  ),

  div(
    class = "subtitle",
    paste0(
      "Prediction of LRI after successful air-enema reduction. ",
      "The classification threshold was fixed during model development."
    )
  ),

  fluidRow(

    column(
      width = 5,

      wellPanel(

        h4("Patient characteristics"),

        uiOutput("feature_inputs"),

        br(),

        actionButton(
          inputId = "calculate",
          label = "Calculate LRI Risk",
          class = "btn-primary"
        ),

        br(),
        br(),

        actionButton(
          inputId = "reset",
          label = "Reset"
        )
      )
    ),

    column(
      width = 7,

      h4("Prediction result"),

      uiOutput("prediction_result"),

      br(),

      helpText(
        "This calculator is intended to support clinical risk assessment ",
        "and should not replace clinical judgment."
      )
    )
  )
)


# ----------------------------------------------------------------------
# Server
# ----------------------------------------------------------------------

server <- function(input, output, session) {

  # --------------------------------------------------------------------
  # Dynamically generate predictor input boxes
  # --------------------------------------------------------------------

  output$feature_inputs <- renderUI({

    tagList(

      lapply(features, function(f) {

        default_value <- if (
          f %in% names(input_defaults)
        ) {
          input_defaults[[f]]
        } else {
          0
        }

        numericInput(
          inputId = paste0("x__", f),
          label = f,
          value = round(default_value, 3)
        )
      })
    )
  })


  # --------------------------------------------------------------------
  # Reset input values
  # --------------------------------------------------------------------

  observeEvent(input$reset, {

    lapply(features, function(f) {

      default_value <- if (
        f %in% names(input_defaults)
      ) {
        input_defaults[[f]]
      } else {
        0
      }

      updateNumericInput(
        session = session,
        inputId = paste0("x__", f),
        value = round(default_value, 3)
      )
    })
  })


  # --------------------------------------------------------------------
  # Individual prediction
  # --------------------------------------------------------------------

  prediction <- eventReactive(
    input$calculate,
    {

      values <- lapply(
        features,
        function(f) {

          value <- input[[paste0("x__", f)]]

          req(value)

          as.numeric(value)
        }
      )

      names(values) <- features

      patient <- as.data.frame(
        values,
        check.names = FALSE
      )

      # Predicted probability of LRI
      probability <- predict(
        wf,
        new_data = patient,
        type = "prob"
      )$.pred_Event

      probability <- as.numeric(probability)

      risk_group <- ifelse(
        probability >= fixed_cutoff,
        "High risk",
        "Low risk"
      )

      list(
        probability = probability,
        risk_group = risk_group
      )
    }
  )


  # --------------------------------------------------------------------
  # Display prediction result
  # --------------------------------------------------------------------

  output$prediction_result <- renderUI({

    req(prediction())

    p <- prediction()$probability
    group <- prediction()$risk_group

    div(
      class = "result-box",

      h4("Predicted probability of LRI"),

      div(
        class = "probability",
        sprintf("%.1f%%", p * 100)
      ),

      br(),

      tags$p(
        paste0(
          "Pre-specified classification threshold: ",
          sprintf("%.1f%%", fixed_cutoff * 100)
        )
      ),

      hr(),

      h4("Risk classification"),

      div(
        class = ifelse(
          group == "High risk",
          "risk-high",
          "risk-low"
        ),
        group
      )
    )
  })
}


# ----------------------------------------------------------------------
# Launch application
# ----------------------------------------------------------------------

shinyApp(
  ui = ui,
  server = server
)
'

writeLines(
  app_code,
  con = file.path(
    SHINY_APP_DIR,
    "app.R"
  )
)


# ----------------------------------------------------------------------
# 29.4 Check deployment files
# ----------------------------------------------------------------------

cat(
  "\\nShiny application directory created at:",
  normalizePath(SHINY_APP_DIR),
  "\\n"
)

cat("\\nFiles prepared for deployment:\\n")

print(
  list.files(
    SHINY_APP_DIR,
    full.names = FALSE
  )
)


# ----------------------------------------------------------------------
# 29.5 Test the application locally
# ----------------------------------------------------------------------
# IMPORTANT:
# Run this line manually before deployment.
#
# shiny::runApp(SHINY_APP_DIR)


# ======================================================================
# 30. DEPLOY TO SHINYAPPS.IO
# ======================================================================

if (!requireNamespace(
  "rsconnect",
  quietly = TRUE
)) {
  install.packages("rsconnect")
}


# ----------------------------------------------------------------------
# 30.1 FIRST DEPLOYMENT ONLY: configure shinyapps.io account
# ----------------------------------------------------------------------
#
# IMPORTANT:
# Do NOT save your token and secret in the manuscript code.
#
# Log in to shinyapps.io:
#
# Account -> Tokens -> Add Token -> Show
#
# Then copy the rsconnect::setAccountInfo(...) command provided by
# shinyapps.io and run it ONCE in the R console.
#
# Example structure ONLY:
#
# rsconnect::setAccountInfo(
#   name   = "YOUR_ACCOUNT_NAME",
#   token  = "YOUR_TOKEN",
#   secret = "YOUR_SECRET"
# )


# ----------------------------------------------------------------------
# 30.2 Optional: inspect package dependencies before deployment
# ----------------------------------------------------------------------

deployment_dependencies <- rsconnect::appDependencies(
  appDir = SHINY_APP_DIR
)

print(deployment_dependencies)


# ----------------------------------------------------------------------
# 30.3 Deploy application
# ----------------------------------------------------------------------
# IMPORTANT:
# Uncomment the following block only after:
#   1. The app runs successfully locally;
#   2. shinyapps.io account information has been configured;
#   3. appDependencies() shows the required packages.
#
# Replace YOUR_ACCOUNT_NAME with your actual shinyapps.io account name.

# rsconnect::deployApp(
#   appDir = SHINY_APP_DIR,
#   appName = SHINY_APP_NAME,
#   appTitle = "LRI Risk Calculator",
#   account = "YOUR_ACCOUNT_NAME",
#   server = "shinyapps.io",
#   appMode = "shiny",
#   launch.browser = TRUE,
#   forceUpdate = TRUE
# )


# ======================================================================
# 31. DEPLOYMENT COMPLETE
# ======================================================================
#
# After successful deployment, the application URL will usually have
# the following format:
#
# https://YOUR_ACCOUNT_NAME.shinyapps.io/LRI-risk-calculator/
#
# Record the final URL for the manuscript.
# ======================================================================