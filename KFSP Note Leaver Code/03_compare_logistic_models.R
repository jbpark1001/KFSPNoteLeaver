# Compare the eight missingness-driven candidate logistic-regression models.
#
# This script preserves the original model specification, complete-case
# handling, 0.50 classification threshold, forced exclusions, and Nagelkerke
# R-squared calculation. It does not install packages during execution.

required_packages <- c("readxl", "DescTools", "writexl")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0) {
  stop(
    "Missing R package(s): ", paste(missing_packages, collapse = ", "),
    ". Install the versions listed in environment/R-packages.txt."
  )
}

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_argument) != 1) {
  stop("Run this file with Rscript so the project root can be resolved reliably.")
}
script_path <- normalizePath(sub("^--file=", "", script_argument), mustWork = TRUE)
project_root <- dirname(dirname(script_path))

cuts_dir <- file.path(project_root, "derived", "cuts")
output_dir <- file.path(project_root, "outputs", "tables")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

target_var <- "NOTE_EX"
sheet_name <- 1
classification_threshold <- 0.5
cut_sizes <- c(32, 43, 50, 58, 68, 84, 97, 143)
cut_files <- file.path(
  cuts_dir,
  sprintf("Cut_%d_DF_%dColumns.xlsx", seq_along(cut_sizes) - 1, cut_sizes)
)

missing_files <- cut_files[!file.exists(cut_files)]
if (length(missing_files) > 0) {
  stop(
    "Candidate-cut file(s) not found. Run 02_select_predictor_subsets.py first:\n",
    paste(missing_files, collapse = "\n")
  )
}

forced_exclusions <- c(
  "YEAR", "PC_ETC", "PC_SPOUSE", "PC_PARENT", "PC_GRANDPARENT",
  "PC_CHILD", "PC_SIBSHIP", "PC_RELATIVE", "PC_LOVER", "PC_FRIEND",
  "PC_STRANGER"
)

format_p_value <- function(p_value, cutoff = 1e-4) {
  if (is.na(p_value)) return("p = NA")
  if (p_value < cutoff) return("p < .0001")
  sprintf("p = %.4f", p_value)
}

fit_and_summarize <- function(
  path,
  subset_number,
  cut_index,
  column_count,
  target = target_var,
  threshold = classification_threshold,
  drop_also = forced_exclusions
) {
  dataframe <- as.data.frame(readxl::read_excel(path, sheet = sheet_name))
  names(dataframe) <- trimws(names(dataframe))
  dataframe[dataframe == -1] <- NA

  if (!target %in% names(dataframe)) {
    stop("Target '", target, "' is absent from: ", path)
  }
  target_values <- na.omit(unique(dataframe[[target]]))
  if (!all(target_values %in% c(0, 1))) {
    stop("Target '", target, "' must be coded 0/1 in: ", path)
  }
  dataframe[[target]] <- factor(dataframe[[target]], levels = c(0, 1))

  predictors <- setdiff(names(dataframe), target)
  dataframe[predictors] <- lapply(dataframe[predictors], function(value) {
    if (is.factor(value)) value else factor(value)
  })

  single_level_full <- vapply(
    dataframe[predictors],
    function(value) nlevels(droplevels(value)) <= 1,
    logical(1)
  )
  dropped_single_level_full <- names(single_level_full)[single_level_full]
  if (length(dropped_single_level_full) > 0) {
    dataframe <- dataframe[
      setdiff(names(dataframe), dropped_single_level_full)
    ]
  }

  predictors <- setdiff(names(dataframe), target)
  if (length(predictors) == 0) stop("No predictors remain in: ", path)
  provisional_formula <- as.formula(
    paste(target, "~", paste(sprintf("`%s`", predictors), collapse = " + "))
  )
  provisional_frame <- model.frame(
    provisional_formula,
    data = dataframe,
    na.action = na.omit
  )

  complete_case_predictors <- setdiff(names(provisional_frame), target)
  single_level_complete <- vapply(
    complete_case_predictors,
    function(column) {
      is.factor(provisional_frame[[column]]) &&
        nlevels(droplevels(provisional_frame[[column]])) <= 1
    },
    logical(1)
  )
  dropped_complete_case <- unique(c(
    names(single_level_complete)[single_level_complete],
    drop_also
  ))

  all_terms <- attr(terms(provisional_formula), "term.labels")
  kept_terms <- setdiff(all_terms, dropped_complete_case)
  if (length(kept_terms) == 0) {
    stop("All predictors became single-level or were excluded in: ", path)
  }

  model_formula <- reformulate(kept_terms, response = target)
  model_frame <- model.frame(model_formula, data = dataframe, na.action = na.omit)
  model <- glm(model_formula, data = model_frame, family = binomial())

  likelihood_ratio_chisq <- model$null.deviance - model$deviance
  likelihood_ratio_df <- model$df.null - model$df.residual
  likelihood_ratio_p <- pchisq(
    likelihood_ratio_chisq,
    df = likelihood_ratio_df,
    lower.tail = FALSE
  )

  probabilities <- fitted(model)
  predicted <- ifelse(probabilities >= threshold, 1, 0)
  truth <- as.numeric(model_frame[[target]]) - 1
  true_positive <- sum(predicted == 1 & truth == 1)
  true_negative <- sum(predicted == 0 & truth == 0)
  false_positive <- sum(predicted == 1 & truth == 0)
  false_negative <- sum(predicted == 0 & truth == 1)

  total <- true_positive + true_negative + false_positive + false_negative
  accuracy <- 100 * (true_positive + true_negative) / total
  sensitivity <- ifelse(
    true_positive + false_negative > 0,
    100 * true_positive / (true_positive + false_negative),
    NA_real_
  )
  specificity <- ifelse(
    true_negative + false_positive > 0,
    100 * true_negative / (true_negative + false_positive),
    NA_real_
  )
  nagelkerke_r2 <- as.numeric(
    DescTools::PseudoR2(model, which = "Nagelkerke")
  )

  actual_n0 <- sum(truth == 0)
  actual_n1 <- sum(truth == 1)
  predicted_n0 <- sum(predicted == 0)
  predicted_n1 <- sum(predicted == 1)

  cat("\n=============================\n")
  cat(sprintf("Subset %d / Cut_%d — %s\n", subset_number, cut_index, basename(path)))
  cat("=============================\n")
  cat(sprintf(
    paste0(
      "Model chi-square = %.2f (%d df), %s; Accuracy = %.1f%%; ",
      "Sensitivity = %.1f%%; Specificity = %.1f%%; Nagelkerke R2 = %.3f; ",
      "N = %d.\n"
    ),
    likelihood_ratio_chisq,
    likelihood_ratio_df,
    format_p_value(likelihood_ratio_p),
    accuracy,
    sensitivity,
    specificity,
    nagelkerke_r2,
    total
  ))

  data.frame(
    manuscript_subset = subset_number,
    cut_index = cut_index,
    column_count = column_count,
    filename = basename(path),
    analysis_n = total,
    chi_square = likelihood_ratio_chisq,
    df = likelihood_ratio_df,
    p_value = likelihood_ratio_p,
    accuracy = accuracy,
    sensitivity = sensitivity,
    specificity = specificity,
    nagelkerke_R2 = nagelkerke_r2,
    actual_n0 = actual_n0,
    actual_n1 = actual_n1,
    predicted_n0 = predicted_n0,
    predicted_n1 = predicted_n1,
    threshold = threshold,
    dropped_single_level_full = paste(dropped_single_level_full, collapse = ";"),
    dropped_or_excluded_complete_case = paste(dropped_complete_case, collapse = ";"),
    stringsAsFactors = FALSE
  )
}

results <- do.call(
  rbind,
  lapply(seq_along(cut_files), function(index) {
    fit_and_summarize(
      path = cut_files[index],
      subset_number = index,
      cut_index = index - 1,
      column_count = cut_sizes[index]
    )
  })
)

csv_path <- file.path(output_dir, "model_fit_summary.csv")
xlsx_path <- file.path(output_dir, "model_fit_summary.xlsx")
write.csv(results, csv_path, row.names = FALSE)
writexl::write_xlsx(results, xlsx_path)

cat("\nSaved model-comparison summaries:\n")
cat(" - ", csv_path, "\n", sep = "")
cat(" - ", xlsx_path, "\n", sep = "")
cat(
  "Selected manuscript model: Subset 3 = Cut_2_DF_50Columns.xlsx.\n"
)
