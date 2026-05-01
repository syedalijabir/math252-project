
# Report/R/compare_results_helpers.R

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(knitr)
  library(mclust)
  library(purrr)
  library(scales)
  library(stringr)
  library(tidyr)
})

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

report_theme <- function() {
  theme_minimal(base_size = 12) +
    theme(
      plot.title = element_text(face = "bold", size = 13, hjust = 0.5),
      plot.subtitle = element_text(size = 10, hjust = 0.5),
      axis.title = element_text(face = "bold"),
      strip.text = element_text(face = "bold"),
      legend.title = element_text(face = "bold"),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      plot.margin = margin(10, 12, 10, 12)
    )
}

blank_plot <- function(title, subtitle = NULL) {
  ggplot() +
    theme_void() +
    labs(title = title, subtitle = subtitle)
}

resolve_cache_sources <- function(cache_root = file.path("..", "output", "benchmark_cache")) {
  candidates <- c(
    cache_root,
    file.path("..", "output", "kmeans_cache"),
    file.path("..", "output", "pd_cache"),
    file.path("..", "output", "rkm_cache"),
    file.path("..", "output", "fpdc_cache")
  )
  
  existing <- unique(candidates[dir.exists(candidates)])
  
  if (length(existing) == 0) {
    stop(
      paste(
        "No cache directories were found.",
        "Expected either ../output/benchmark_cache or the per-method cache folders under ../output/."
      )
    )
  }
  
  normalizePath(existing, winslash = "/", mustWork = FALSE)
}

parse_model_filename <- function(filename) {
  patterns <- list(
    KMeans = "^kmeans_(user_embeddings_([0-9]+))_k([0-9]+)\\.rds$",
    PD     = "^pd_(user_embeddings_([0-9]+))_k([0-9]+)\\.rds$",
    RKM    = "^rkm_(user_embeddings_([0-9]+))_k([0-9]+)_(?:q|nf)([0-9]+)\\.rds$",
    FPDC   = "^fpdc_(user_embeddings_([0-9]+))_k([0-9]+)_(?:q|nf)([0-9]+)(?:_nu([0-9]+))?\\.rds$"
  )
  
  if (str_detect(filename, patterns$KMeans)) {
    m <- str_match(filename, patterns$KMeans)
    return(tibble(
      method = "KMeans",
      embedding = m[, 2],
      embedding_dim = as.integer(m[, 3]),
      k = as.integer(m[, 4]),
      q = NA_integer_,
      nu = NA_integer_
    ))
  }
  
  if (str_detect(filename, patterns$PD)) {
    m <- str_match(filename, patterns$PD)
    return(tibble(
      method = "PD",
      embedding = m[, 2],
      embedding_dim = as.integer(m[, 3]),
      k = as.integer(m[, 4]),
      q = NA_integer_,
      nu = NA_integer_
    ))
  }
  
  if (str_detect(filename, patterns$RKM)) {
    m <- str_match(filename, patterns$RKM)
    return(tibble(
      method = "RKM",
      embedding = m[, 2],
      embedding_dim = as.integer(m[, 3]),
      k = as.integer(m[, 4]),
      q = as.integer(m[, 5]),
      nu = NA_integer_
    ))
  }
  
  if (str_detect(filename, patterns$FPDC)) {
    m <- str_match(filename, patterns$FPDC)
    parsed_nu <- suppressWarnings(as.integer(m[, 6]))
    return(tibble(
      method = "FPDC",
      embedding = m[, 2],
      embedding_dim = as.integer(m[, 3]),
      k = as.integer(m[, 4]),
      q = as.integer(m[, 5]),
      nu = ifelse(is.na(parsed_nu), 10L, parsed_nu)
    ))
  }
  
  tibble(
    method = NA_character_,
    embedding = NA_character_,
    embedding_dim = NA_integer_,
    k = NA_integer_,
    q = NA_integer_,
    nu = NA_integer_
  )
}

list_model_files <- function(cache_root = file.path("..", "output", "benchmark_cache")) {
  all_files <- unlist(
    lapply(
      resolve_cache_sources(cache_root),
      function(path) list.files(path, pattern = "\\.rds$", recursive = TRUE, full.names = TRUE)
    ),
    use.names = FALSE
  )
  
  tibble(file = sort(unique(all_files))) %>%
    mutate(filename = basename(file)) %>%
    mutate(meta = purrr::map(filename, parse_model_filename)) %>%
    tidyr::unnest(meta) %>%
    filter(!is.na(method)) %>%
    select(file, method, embedding, embedding_dim, k, q, nu)
}

find_value <- function(obj, candidate_names) {
  locations <- list(obj)
  
  if (is.list(obj) && "model" %in% names(obj) && !is.null(obj$model)) {
    locations <- c(locations, list(obj$model))
  }
  
  for (loc in locations) {
    if (is.null(loc) || !is.list(loc)) next
    
    for (nm in candidate_names) {
      if (nm %in% names(loc) && !is.null(loc[[nm]])) {
        return(loc[[nm]])
      }
    }
  }
  
  NULL
}

coerce_num <- function(x) {
  if (is.null(x) || length(x) == 0) return(NA_real_)
  suppressWarnings(as.numeric(x)[1])
}

extract_labels <- function(obj) {
  direct_labels <- find_value(obj, c("labels", "label", "cluster", "clustering"))
  if (!is.null(direct_labels)) {
    return(as.integer(unlist(direct_labels)))
  }
  
  posterior <- find_value(obj, c("probabilities", "posterior", "post", "u"))
  if (is.matrix(posterior) || is.data.frame(posterior)) {
    return(max.col(as.matrix(posterior), ties.method = "first"))
  }
  
  NULL
}

extract_runtime <- function(obj) {
  coerce_num(find_value(obj, c("runtime_sec", "runtime_total_sec", "runtime", "elapsed_sec", "elapsed")))
}

extract_silhouette <- function(obj) {
  coerce_num(find_value(obj, c("Silhouette", "avg_silhouette", "silhouette", "avg_width")))
}

extract_ch <- function(obj) {
  coerce_num(find_value(obj, c("CH", "ch", "calinski_harabasz")))
}

extract_prob_silhouette <- function(obj) {
  coerce_num(find_value(obj, c("ProbSilh", "prob_silh", "probability_silhouette")))
}

read_model_record <- function(file, method, embedding, embedding_dim, k, q, nu) {
  obj <- tryCatch(readRDS(file), error = function(e) NULL)
  
  if (is.null(obj)) {
    return(tibble(
      file = file,
      method = method,
      embedding = embedding,
      embedding_dim = embedding_dim,
      k = k,
      q = q,
      nu = nu,
      runtime_sec = NA_real_,
      silhouette = NA_real_,
      ch = NA_real_,
      prob_silh = NA_real_,
      n_obs = NA_integer_,
      status = "read_error",
      labels = list(NULL)
    ))
  }
  
  labels <- extract_labels(obj)
  
  tibble(
    file = file,
    method = method,
    embedding = embedding,
    embedding_dim = embedding_dim,
    k = k,
    q = q,
    nu = nu,
    runtime_sec = extract_runtime(obj),
    silhouette = extract_silhouette(obj),
    ch = extract_ch(obj),
    prob_silh = extract_prob_silhouette(obj),
    n_obs = if (is.null(labels)) NA_integer_ else length(labels),
    status = "ok",
    labels = list(labels)
  )
}

compute_ari <- function(metrics_df) {
  usable <- metrics_df %>%
    filter(status == "ok", !is.na(k), !purrr::map_lgl(labels, is.null))
  
  if (nrow(usable) < 2) {
    return(tibble())
  }
  
  ari_rows <- list()
  counter <- 1
  
  for (emb in sort(unique(usable$embedding_dim))) {
    current_emb <- usable %>% filter(embedding_dim == emb)
    
    for (k_val in sort(unique(current_emb$k))) {
      current <- current_emb %>% filter(k == k_val)
      if (nrow(current) < 2) next
      
      for (i in seq_len(nrow(current) - 1)) {
        for (j in (i + 1):nrow(current)) {
          labels_i <- current$labels[[i]]
          labels_j <- current$labels[[j]]
          
          if (is.null(labels_i) || is.null(labels_j)) next
          if (length(labels_i) != length(labels_j)) next
          
          ari_rows[[counter]] <- tibble(
            embedding_dim = emb,
            k = k_val,
            method_1 = as.character(current$method[i]),
            method_2 = as.character(current$method[j]),
            q_1 = current$q[i],
            q_2 = current$q[j],
            nu_1 = current$nu[i],
            nu_2 = current$nu[j],
            ari = tryCatch(
              mclust::adjustedRandIndex(labels_i, labels_j),
              error = function(e) NA_real_
            )
          )
          counter <- counter + 1
        }
      }
    }
  }
  
  bind_rows(ari_rows)
}

summarise_ari <- function(ari_df) {
  if (nrow(ari_df) == 0) {
    return(tibble())
  }
  
  ari_df %>%
    mutate(method_pair = paste(pmin(method_1, method_2), pmax(method_1, method_2), sep = "_vs_")) %>%
    group_by(method_pair) %>%
    summarise(
      n_comparisons = n(),
      avg_ari = mean(ari, na.rm = TRUE),
      median_ari = median(ari, na.rm = TRUE),
      min_ari = min(ari, na.rm = TRUE),
      max_ari = max(ari, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(desc(avg_ari))
}

format_best_table <- function(best_models_df) {
  best_models_df %>%
    mutate(
      q_display = case_when(
        method %in% c("RKM", "FPDC") & !is.na(q) ~ as.character(q),
        TRUE ~ "—"
      ),
      nu_display = case_when(
        method == "FPDC" & !is.na(nu) ~ as.character(nu),
        TRUE ~ "—"
      ),
      prob_silh_display = case_when(
        !is.na(prob_silh) ~ sprintf("%.3f", prob_silh),
        TRUE ~ "—"
      )
    ) %>%
    transmute(
      Method = as.character(method),
      Embedding = embedding_dim,
      k = k,
      q = q_display,
      nu = nu_display,
      Runtime_sec = round(runtime_sec, 1),
      Silhouette = round(silhouette, 3),
      CH = round(ch, 2),
      ProbSilh = prob_silh_display
    )
}

format_ari_summary_table <- function(ari_summary) {
  if (nrow(ari_summary) == 0) {
    return(tibble(Note = "No pairwise ARI results were available from the cached models."))
  }
  
  ari_summary %>%
    transmute(
      Method_Pair = str_replace_all(method_pair, "_vs_", " vs "),
      Comparisons = n_comparisons,
      Average_ARI = round(avg_ari, 3),
      Median_ARI = round(median_ari, 3),
      Min_ARI = round(min_ari, 3),
      Max_ARI = round(max_ari, 3)
    )
}

build_compare_results_data <- function(
    cache_root = file.path("..", "output", "benchmark_cache"),
    save_csv = FALSE,
    output_dir = file.path("figures", "compare_results")
) {
  meta <- list_model_files(cache_root)
  
  if (nrow(meta) == 0) {
    stop("No per-model .rds files matched the expected naming patterns.")
  }
  
  metrics_df <- purrr::pmap_dfr(meta, read_model_record) %>%
    mutate(
      method = factor(method, levels = c("KMeans", "PD", "RKM", "FPDC")),
      embedding_label = factor(
        embedding_dim,
        levels = sort(unique(embedding_dim)),
        labels = paste0(sort(unique(embedding_dim)), "-dimensional")
      ),
      model_id = paste0(
        method, "__",
        embedding, "__k", k,
        ifelse(is.na(q), "", paste0("__q", q)),
        ifelse(is.na(nu), "", paste0("__nu", nu))
      )
    )
  
  plot_df <- metrics_df %>%
    arrange(method, embedding_dim, k, q, nu)
  
  best_models_df <- plot_df %>%
    filter(!is.na(silhouette)) %>%
    group_by(method, embedding_dim, embedding_label) %>%
    arrange(desc(silhouette), desc(ch), .by_group = TRUE) %>%
    slice(1) %>%
    ungroup() %>%
    arrange(embedding_dim, method)
  
  if (nrow(best_models_df) == 0) {
    stop("No valid silhouette scores were found in the cached results.")
  }
  
  best_table <- format_best_table(best_models_df)
  overall_best <- best_models_df %>% arrange(desc(silhouette), desc(ch)) %>% slice(1)
  ari_df <- compute_ari(plot_df)
  ari_summary <- summarise_ari(ari_df)
  
  if (isTRUE(save_csv)) {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    
    write.csv(plot_df %>% select(-labels), file.path(output_dir, "plot_ready_metrics.csv"), row.names = FALSE)
    write.csv(best_models_df %>% select(-labels), file.path(output_dir, "best_models_numeric.csv"), row.names = FALSE)
    write.csv(best_table, file.path(output_dir, "best_models_report_table.csv"), row.names = FALSE)
    
    if (nrow(ari_df) > 0) {
      write.csv(ari_df, file.path(output_dir, "ari_pairwise_same_embedding_same_k.csv"), row.names = FALSE)
    }
    
    if (nrow(ari_summary) > 0) {
      write.csv(ari_summary, file.path(output_dir, "ari_summary_by_method_pair.csv"), row.names = FALSE)
    }
  }
  
  list(
    meta = meta,
    metrics_df = metrics_df,
    plot_df = plot_df,
    best_models_df = best_models_df,
    best_table = best_table,
    overall_best = overall_best,
    ari_df = ari_df,
    ari_summary = ari_summary
  )
}

plot_best_silhouette_by_method_embedding <- function(plot_df) {
  fig_df <- plot_df %>%
    filter(!is.na(silhouette)) %>%
    group_by(method, embedding_label) %>%
    summarise(best_silhouette = max(silhouette, na.rm = TRUE), .groups = "drop")
  
  ggplot(fig_df, aes(x = embedding_label, y = best_silhouette, fill = method)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.65) +
    labs(
      title = "Best Silhouette Score by Method and Embedding Dimension",
      x = "Embedding dimension",
      y = "Best average silhouette width",
      fill = "Method"
    ) +
    report_theme()
}

plot_runtime_vs_silhouette <- function(plot_df) {
  fig_df <- plot_df %>% filter(!is.na(runtime_sec), !is.na(silhouette))
  
  if (nrow(fig_df) == 0) {
    return(blank_plot("Runtime-performance plot unavailable"))
  }
  
  ggplot(fig_df, aes(x = runtime_sec, y = silhouette, color = method, shape = method)) +
    geom_point(size = 2.6, alpha = 0.85) +
    scale_x_log10(labels = scales::label_number()) +
    labs(
      title = "Runtime-Performance Tradeoff",
      subtitle = "Lower runtime and higher silhouette are preferred",
      x = "Runtime (seconds, log scale)",
      y = "Average silhouette width",
      color = "Method",
      shape = "Method"
    ) +
    report_theme()
}

plot_ari_method_pairs <- function(ari_df) {
  if (nrow(ari_df) == 0) {
    return(blank_plot("Average ARI by method pair unavailable"))
  }
  
  fig_df <- ari_df %>%
    filter(method_1 != method_2) %>%
    mutate(method_pair = paste(pmin(method_1, method_2), pmax(method_1, method_2), sep = " vs ")) %>%
    group_by(method_pair) %>%
    summarise(avg_ari = mean(ari, na.rm = TRUE), .groups = "drop") %>%
    arrange(desc(avg_ari))
  
  ggplot(fig_df, aes(x = reorder(method_pair, avg_ari), y = avg_ari)) +
    geom_col(width = 0.7) +
    coord_flip() +
    labs(
      title = "Average Adjusted Rand Index by Method Pair",
      subtitle = "Higher ARI indicates stronger agreement between clustering solutions",
      x = "Method pair",
      y = "Average ARI"
    ) +
    report_theme()
}

plot_silhouette_vs_k <- function(plot_df, methods = c("KMeans", "PD")) {
  fig_df <- plot_df %>%
    filter(method %in% methods, !is.na(silhouette)) %>%
    select(method, embedding_label, k, silhouette)
  
  if (nrow(fig_df) == 0) {
    return(blank_plot("Silhouette-by-k plot unavailable"))
  }
  
  ggplot(fig_df, aes(x = k, y = silhouette, color = method, group = method)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 2) +
    facet_wrap(~ embedding_label, nrow = 1) +
    scale_x_continuous(breaks = sort(unique(fig_df$k))) +
    labs(
      title = "Silhouette Scores Across k",
      subtitle = paste(paste(methods, collapse = " and "), "across embedding dimensions"),
      x = "Number of clusters (k)",
      y = "Average silhouette width",
      color = "Method"
    ) +
    report_theme()
}

plot_rkm_sensitivity <- function(plot_df) {
  fig_df <- plot_df %>%
    filter(method == "RKM", !is.na(q), !is.na(silhouette))
  
  if (nrow(fig_df) == 0) {
    return(blank_plot("RKM tuning plot unavailable"))
  }
  
  ggplot(fig_df, aes(x = k, y = silhouette, color = factor(q), group = q)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 2) +
    facet_wrap(~ embedding_label, nrow = 1) +
    scale_x_continuous(breaks = sort(unique(fig_df$k))) +
    labs(
      title = "Reduced K-means: Silhouette Scores Across k and q",
      subtitle = "Supporting diagnostic for the reduced K-means tuning grid",
      x = "Number of clusters (k)",
      y = "Average silhouette width",
      color = "q"
    ) +
    report_theme()
}

plot_prob_silhouette <- function(plot_df) {
  fig_df <- plot_df %>%
    filter(method %in% c("PD", "FPDC"), !is.na(prob_silh))
  
  if (nrow(fig_df) == 0) {
    return(blank_plot("Probability silhouette plot unavailable"))
  }
  
  ggplot(fig_df, aes(x = k, y = prob_silh, color = method, group = interaction(method, q, nu))) +
    geom_line(alpha = 0.7, linewidth = 0.8) +
    geom_point(size = 1.8) +
    facet_wrap(~ embedding_label, nrow = 1) +
    scale_x_continuous(breaks = sort(unique(fig_df$k))) +
    labs(
      title = "Probability Silhouette for PD and FPDC",
      subtitle = "Supporting diagnostic for the probability-based methods",
      x = "Number of clusters (k)",
      y = "Probability silhouette",
      color = "Method"
    ) +
    report_theme()
}