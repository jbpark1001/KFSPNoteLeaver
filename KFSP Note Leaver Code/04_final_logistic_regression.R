# Fit the selected demographic model and export inferential tables.
#
# The manuscript's Subset 3 is the zero-indexed file Cut_2 (50 columns). The
# model specification, reference coding, complete-case handling, exclusions,
# factor tests, odds ratios, confidence intervals, and significance filters are
# unchanged from the original analysis script.

required_packages <- c("readxl", "dplyr", "broom", "tibble", "writexl")
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

selected_subset <- 3
selected_cut_index <- 2
selected_column_count <- 50
target_var <- "NOTE_EX"
sheet_name <- 1

data_path <- file.path(
  project_root,
  "derived",
  "cuts",
  "Cut_2_DF_50Columns.xlsx"
)
output_dir <- file.path(project_root, "outputs", "tables")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(data_path)) {
  stop(
    "Selected candidate cut not found: ", data_path,
    "\nRun 02_select_predictor_subsets.py first."
  )
}

dataframe <- as.data.frame(readxl::read_excel(data_path, sheet = sheet_name))
names(dataframe) <- trimws(names(dataframe))
dataframe[dataframe == -1] <- NA

if (!target_var %in% names(dataframe)) {
  stop("Target '", target_var, "' is absent from: ", data_path)
}
target_values <- na.omit(unique(dataframe[[target_var]]))
if (!all(target_values %in% c(0, 1))) {
  stop("Target must be coded as 0/1. Please check preprocessing.")
}
dataframe[[target_var]] <- factor(dataframe[[target_var]], levels = c(0, 1))

predictors <- setdiff(names(dataframe), target_var)
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
  dataframe <- dplyr::select(
    dataframe,
    -dplyr::all_of(dropped_single_level_full)
  )
}

predictors <- setdiff(names(dataframe), target_var)
if (length(predictors) == 0) stop("No predictors remain after preprocessing.")
provisional_formula <- as.formula(
  paste(target_var, "~", paste(sprintf("`%s`", predictors), collapse = " + "))
)
provisional_frame <- model.frame(
  provisional_formula,
  data = dataframe,
  na.action = na.omit
)

complete_case_predictors <- setdiff(names(provisional_frame), target_var)
single_level_complete <- vapply(
  complete_case_predictors,
  function(column) {
    is.factor(provisional_frame[[column]]) &&
      nlevels(droplevels(provisional_frame[[column]])) <= 1
  },
  logical(1)
)

forced_exclusions <- c(
  "YEAR", "PC_ETC", "PC_SPOUSE", "PC_PARENT", "PC_GRANDPARENT",
  "PC_CHILD", "PC_SIBSHIP", "PC_RELATIVE", "PC_LOVER", "PC_FRIEND",
  "PC_STRANGER"
)
dropped_complete_case <- unique(c(
  names(single_level_complete)[single_level_complete],
  forced_exclusions
))

all_terms <- attr(terms(provisional_formula), "term.labels")
kept_terms <- setdiff(all_terms, dropped_complete_case)
if (length(kept_terms) == 0) {
  stop("All predictors became single-level or were explicitly excluded.")
}

model_formula <- reformulate(kept_terms, response = target_var)
model_frame <- model.frame(model_formula, data = dataframe, na.action = na.omit)
glm_fit <- glm(model_formula, data = model_frame, family = binomial())

# Factor-level likelihood-ratio tests, matching the original drop1 workflow.
var_tests <- drop1(glm_fit, test = "Chisq") |>
  tibble::rownames_to_column("Variable") |>
  dplyr::rename(Factor_p_value = `Pr(>Chi)`)

# Level coefficients, exponentiated as odds ratios with the original CI call.
or_table <- broom::tidy(
  glm_fit,
  exponentiate = TRUE,
  conf.int = TRUE,
  conf.method = "wald"
) |>
  dplyr::filter(term != "(Intercept)") |>
  dplyr::mutate(
    Variable = sub("^(.*?)[:].*$", "\\1", term),
    Variable = sub("^(.*?)\\..*$", "\\1", Variable),
    Variable = sub("^(.*?)\\d.*$", "\\1", Variable)
  )

combined_table <- var_tests |>
  dplyr::left_join(or_table, by = "Variable") |>
  dplyr::arrange(Factor_p_value, p.value)

filtered_table <- combined_table |>
  dplyr::filter(Factor_p_value < 0.05) |>
  dplyr::filter(is.na(p.value) | p.value < 0.05) |>
  dplyr::arrange(Factor_p_value, Variable, p.value)

anova_table <- anova(glm_fit, test = "Chisq") |>
  as.data.frame() |>
  tibble::rownames_to_column("Variable")

analysis_metadata <- data.frame(
  manuscript_subset = selected_subset,
  cut_index = selected_cut_index,
  column_count = selected_column_count,
  source_file = basename(data_path),
  analysis_n = nrow(model_frame),
  outcome_0_n = sum(model_frame[[target_var]] == "0"),
  outcome_1_n = sum(model_frame[[target_var]] == "1"),
  dropped_single_level_full = paste(dropped_single_level_full, collapse = ";"),
  dropped_or_excluded_complete_case = paste(dropped_complete_case, collapse = ";"),
  stringsAsFactors = FALSE
)

base_name <- "Cut_2_DF_50Columns"
writexl::write_xlsx(
  filtered_table,
  file.path(output_dir, paste0("Filtered_", base_name, ".xlsx"))
)
writexl::write_xlsx(
  combined_table,
  file.path(output_dir, paste0("combined_table_", base_name, ".xlsx"))
)
writexl::write_xlsx(
  var_tests,
  file.path(output_dir, paste0("var_tests_", base_name, ".xlsx"))
)
writexl::write_xlsx(
  or_table,
  file.path(output_dir, paste0("or_table_", base_name, ".xlsx"))
)
writexl::write_xlsx(
  anova_table,
  file.path(output_dir, paste0("anova_", base_name, ".xlsx"))
)
write.csv(
  analysis_metadata,
  file.path(output_dir, "selected_model_metadata.csv"),
  row.names = FALSE
)

summary_path <- file.path(output_dir, "selected_model_summary.txt")
capture.output(
  {
    cat("Selected model: manuscript Subset 3 = Cut_2 (50 columns)\n\n")
    print(summary(glm_fit))
    cat("\nLikelihood-ratio analysis of deviance\n")
    print(anova(glm_fit, test = "Chisq"))
  },
  file = summary_path
)

cat(
  sprintf(
    "Fitted manuscript Subset 3 / Cut_2 with N = %s complete cases.\n",
    format(nrow(model_frame), big.mark = ",")
  )
)
cat("Saved final-model tables to: ", output_dir, "\n", sep = "")
