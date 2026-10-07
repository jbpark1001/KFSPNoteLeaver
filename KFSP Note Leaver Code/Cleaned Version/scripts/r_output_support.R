# Export through installed pandas/openpyxl when writexl is unavailable.
# This changes file packaging only; all models and dataframes stay in R.
write_analysis_xlsx <- function(data, path) {
  if (requireNamespace("writexl", quietly=TRUE)) {
    writexl::write_xlsx(data, path)
  } else {
    csv <- sub("\\.xlsx$", ".csv", path)
    write.csv(data, csv, row.names=FALSE, na="")
    python <- Sys.getenv("MANUSCRIPT_PYTHON", "python")
    bridge <- file.path(project_root, "tools", "csv_to_xlsx.py")
    status <- system2(python, c(shQuote(bridge), shQuote(csv), shQuote(path)))
    if (status != 0L || !file.exists(path)) stop("Excel export failed: ", path)
  }
}

nagelkerke_r2 <- function(model) {
  if (requireNamespace("DescTools", quietly=TRUE)) {
    return(as.numeric(DescTools::PseudoR2(model, which="Nagelkerke")))
  }
  # Identical likelihood formula for the unweighted Bernoulli GLMs in this study.
  n <- nobs(model)
  (1-exp(-(model$null.deviance-model$deviance)/n)) /
    (1-exp(-model$null.deviance/n))
}
