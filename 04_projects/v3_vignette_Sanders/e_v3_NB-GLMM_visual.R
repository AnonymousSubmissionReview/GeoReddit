# ============================================================
# Purpose:
#   - Analyze daily-level Reddit activity (posts / unique users) around U.S.
#     state primary election dates.
#   - Construct ±N-day windows (N ∈ {3,7,15,30,60,90}), fit NB-GLMMs (glmmTMB; 
#     log link, state random intercept), and extract fixed-effect 
#     IRRs (Pre vs Primary, Post vs Primary) with 95% CIs and p-values.
#   - Produce summary tables and visualizations (forest plots & event-aligned
#     time series with baseline normalization -90 to -8 days).
# 
# Methods implemented:
#   - Negative Binomial GLMMs with state random intercepts.
#   - IRRs for Pre vs. Primary and Post vs. Primary.
#   - Event-aligned time series with baseline normalization (-90 to -8 days).
#
# Notes:
#   - Model: value ~ group + (1|state), family = nbinom2(link="log")
#   - Groups: Average Pre / Primary Day / Average Post (reference = Primary Day)
#   - Windows: ±3, ±7, ±15, ±30, ±60, ±90 days
#
# Input:
#   1) p_user_type.csv
#        Columns: time, author, type, num
#        Daily user/post activity (from r/SandersForPresident
#                                  2015.11-2016.9 and 2019.11-2020.7)
#   2) o_2005-06to2023-12_filtered_authors.csv
#        Columns: author, state
#        Maps GeoReddit authors to U.S. states. (Ratio2>1)
#   3) o_primary_dates.csv
#        Columns: state, date, year
#        Primary election dates by state and year (2016 and 2020).
#。      Example:
#       state,date,year
#       IA,2016-02-01,2016
#       NH,2016-02-09,2016
#       IA,2020-02-03,2020
#       NH,2020-02-11,2020
#
# Output:
#   Tables:
#     - p_summary_users_daily.csv   (IRRs, 95%CI and p value for unique users)
#     - p_summary_posts_daily.csv   (IRRs, 95%CI and p value for posts)
#
#   Figures:
#     - figs/p_forest_daily_7d.png      (main ±7d IRR plot)
#     - figs/p_timeseries_daily_90d.png (event-aligned ±90d plot)
#     - figs/p_forest_daily_users.png   (forest plots for users, all windows)
#     - figs/p_forest_daily_posts.png   (forest plots for posts, all windows)
#     - figs/p_timeseries_daily_users.png / posts.png (event-aligned plots, robustness)
#
# Example usage:
#   Rscript e_v3_NB-GLMM_visual.R \
#     --input_folder "/path/to/input_folder" \
#     --output_folder "/path/to/output_folder"
#
# ============================================================
suppressPackageStartupMessages({
  library(tidyverse)
  library(lubridate)
  library(glmmTMB)
  library(broom.mixed)
  library(janitor)
  library(readr)
  library(ggplot2)
  library(stringr)
  library(scales)
  library(purrr)
  library(forcats)
  library(optparse)
})

# ===================== COMMAND LINE ARGUMENTS =====================
option_list <- list(
  make_option(c("--input_folder"), 
              type = "character", 
              default = NULL, 
              help = "Input folder path containing required CSV files", 
              metavar = "character"),
  make_option(c("--output_folder"), 
              type = "character", 
              default = NULL, 
              help = "Output folder path for results", 
              metavar = "character")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

# Check required parameters
if (is.null(opt$input_folder) || is.null(opt$output_folder)) {
  print_help(opt_parser)
  stop("Both --input_folder and --output_folder arguments are required", call. = FALSE)
}

# Check if input folder exists
if (!dir.exists(opt$input_folder)) {
  stop("Input folder does not exist: ", opt$input_folder, call. = FALSE)
}

# ===================== FILE PATHS SETUP =====================
input_folder  <- normalizePath(opt$input_folder)
output_folder <- normalizePath(opt$output_folder, mustWork = FALSE)

# Input file paths (assuming fixed file names)
hist_csv    <- file.path(input_folder, "p_user_type.csv")                           # time, author, type, num
authors_csv <- file.path(input_folder, "o_2005-06to2023-12_filtered_authors.csv")  # author, state
primary_csv <- file.path(input_folder, "o_primary_dates.csv")                      # state,date,year

# Check if required input files exist
required_files <- c(hist_csv, authors_csv, primary_csv)
missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0) {
  stop("The following required files are missing:\n", paste(missing_files, collapse = "\n"), call. = FALSE)
}

# Create output directory if it doesn't exist
ensure_dir <- function(p) if (!dir.exists(p)) dir.create(p, recursive = TRUE, showWarnings = FALSE)
ensure_dir(output_folder)

message("Input folder: ", input_folder)
message("Output folder: ", output_folder)

# ===================== DATA LOADING AND CLEANING =====================
message("Loading data files...")

hist <- read_csv(hist_csv, show_col_types = FALSE) %>%
  clean_names() %>% mutate(time = as.Date(time))

authors <- read_csv(authors_csv, show_col_types = FALSE) %>%
  clean_names() %>% mutate(state = str_to_upper(str_trim(state)))

primary <- read_csv(primary_csv, show_col_types = FALSE) %>%
  clean_names() %>% mutate(state = str_to_upper(str_trim(state)),
                           date  = as.Date(date),
                           year  = as.integer(year))

message("Data loading completed.")

# ===================== DATA AGGREGATION (Daily level) =====================
# Join historical data with author state information
dat <- hist %>%
  inner_join(authors %>% select(author, state), by = "author") %>%
  mutate(year = year(time))

# Aggregate daily posts by state
daily_posts <- dat %>%
  group_by(state, date = time) %>%
  summarise(posts = sum(num, na.rm = TRUE), .groups = "drop")

# Aggregate daily unique users by state
daily_users <- dat %>%
  distinct(state, date = time, author) %>%
  count(state, date, name = "users")

# ===================== WINDOW DATA CONSTRUCTION =====================
# Create window data for daily analysis
# @param measure: "users" or "posts"
# @param year_sel: selected year for analysis
# @param N_days: number of days for pre/post window
# @return: data frame with pre/primary/post periods
make_daily_window_df <- function(measure = c("users", "posts"), year_sel, N_days) {
  measure <- match.arg(measure)
  base_df <- if (measure == "users") daily_users else daily_posts
  pri_y   <- primary %>% filter(year == year_sel)
  
  if (nrow(pri_y) == 0) return(tibble())
  
  purrr::map_dfr(seq_len(nrow(pri_y)), function(i) {
    st <- pri_y$state[i]; pday <- pri_y$date[i]
    g  <- base_df %>% filter(state == st) %>% arrange(date)
    
    if (nrow(g) == 0) return(tibble())
    
    pre_rng  <- seq(pday - days(N_days), pday - days(1), by = "1 day")
    post_rng <- seq(pday + days(1),     pday + days(N_days), by = "1 day")
    
    bind_rows(
      g %>% filter(date %in% pre_rng)  %>% mutate(group = "Average Pre"),
      g %>% filter(date == pday)       %>% mutate(group = "Primary Day"),
      g %>% filter(date %in% post_rng) %>% mutate(group = "Average Post")
    ) %>% transmute(state, year = year_sel, group, value = .data[[measure]])
  }) %>%
    mutate(group = factor(group, levels = c("Average Pre","Primary Day","Average Post")),
           state = factor(state))
}

# ===================== MODEL FITTING AND EXTRACTION =====================
# Fit negative binomial GLMM model
# @param df: input data frame
# @return: fitted glmmTMB model or NULL if fitting fails
fit_nb_glmm <- function(df) {
  if (nrow(df) == 0 || dplyr::n_distinct(df$state) < 2) return(NULL)
  
  df <- df %>% mutate(group = forcats::fct_relevel(group, "Primary Day"))
  ctrl <- glmmTMBControl(optCtrl = list(iter.max = 1000, eval.max = 1000))
  
  glmmTMB(value ~ group + (1|state), family = nbinom2(link = "log"), data = df, control = ctrl)
}

# Extract fixed effects from fitted model
# @param fit: fitted glmmTMB model
# @return: data frame with fixed effects and convergence info
extract_fixed <- function(fit) {
  if (is.null(fit)) return(tibble())
  
  fx_log <- broom.mixed::tidy(fit, effects = "fixed", conf.int = TRUE, conf.level = 0.95)
  fx_rr  <- broom.mixed::tidy(fit, effects = "fixed", conf.int = TRUE, conf.level = 0.95, exponentiate = TRUE)
  
  fx_log %>%
    select(term, estimate, std.error, statistic, p.value, conf.low, conf.high) %>%
    rename(log_estimate = estimate, log_se = std.error, log_stat = statistic, log_p = p.value,
           log_low = conf.low, log_up = conf.high) %>%
    left_join(
      fx_rr %>% select(term, estimate, conf.low, conf.high) %>%
        rename(rate_ratio = estimate, rr_low = conf.low, rr_up = conf.high),
      by = "term"
    ) %>%
    mutate(converged = isTRUE(fit$sdr$pdHess))
}

# ===================== MAIN ANALYSIS LOOP (Daily only) =====================
message("Starting model fitting...")

years_run <- sort(unique(primary$year))
day_windows  <- c(3,7,15,30,60,90)
fx_all <- list()

for (yy in years_run) {
  message("Processing year: ", yy)
  for (m in c("users","posts")) {
    for (w in day_windows) {
      dfi <- make_daily_window_df(measure = m, year_sel = yy, N_days = w)
      if (nrow(dfi) == 0) next
      
      fit <- fit_nb_glmm(dfi)
      if (is.null(fit)) next
      
      fx  <- extract_fixed(fit) %>%
        filter(term != "(Intercept)") %>%
        mutate(
          term = recode(term,
                        "groupAverage Pre"  = "Pre vs Primary",
                        "groupAverage Post" = "Post vs Primary"),
          year   = yy,
          window = paste0("±", w, "d"),
          target = m
        )
      fx_all[[length(fx_all) + 1]] <- fx
    }
  }
}

fx_all <- dplyr::bind_rows(fx_all)
message("Daily models completed.")

# ===================== SUMMARY TABLES (Daily only) =====================
# Format p-values with significance stars
# @param p: p-value vector
# @return: formatted p-value strings
p_star_fmt <- function(p) {
  case_when(
    is.na(p) ~ NA_character_,
    p < 0.001 ~ "<0.001***",
    p < 0.01  ~ "<0.01**",
    p < 0.05  ~ "<0.05*",
    TRUE ~ sprintf("%.3f", p)
  )
}

# Create summary table for a specific target measure
# @param fx_all: combined results from all models
# @param target_key: "users" or "posts"
# @return: formatted summary table
make_window_table <- function(fx_all, target_key = c("users","posts")) {
  target_key <- match.arg(target_key)
  fx_sub <- fx_all %>%
    filter(target == target_key,
           term %in% c("Pre vs Primary","Post vs Primary")) %>%
    mutate(term_key = recode(term, "Pre vs Primary"="pre", "Post vs Primary"="post"))
  
  conv_df <- fx_sub %>%
    group_by(year, window) %>%
    summarise(converged_any = any(converged, na.rm = TRUE), .groups = "drop")
  
  wide <- fx_sub %>%
    select(year, window, term_key, rate_ratio, rr_low, rr_up, log_p) %>%
    pivot_wider(
      id_cols = c(year, window),
      names_from = term_key,
      values_from = c(rate_ratio, rr_low, rr_up, log_p),
      names_sep = "_"
    ) %>%
    left_join(conv_df, by = c("year","window"))
  
  out <- wide %>%
    transmute(
      year, window,
      `Pre vs Primary (IRR)`   = rate_ratio_pre,
      `Pre vs Primary (p)`     = p_star_fmt(log_p_pre),
      `Pre vs Primary (95%CI)` = ifelse(is.finite(rr_low_pre) & is.finite(rr_up_pre),
                                        sprintf("[%.2f, %.2f]", rr_low_pre, rr_up_pre), NA_character_),
      `Post vs Primary (IRR)`   = rate_ratio_post,
      `Post vs Primary (p)`     = p_star_fmt(log_p_post),
      `Post vs Primary (95%CI)` = ifelse(is.finite(rr_low_post) & is.finite(rr_up_post),
                                         sprintf("[%.2f, %.2f]", rr_low_post, rr_up_post), NA_character_),
      Converged = case_when(is.na(converged_any) ~ NA_character_,
                            converged_any ~ "Yes", TRUE ~ "No")
    ) %>%
    # Sort by window width
    mutate(win_num = suppressWarnings(as.numeric(str_extract(window, "\\d+")))) %>%
    arrange(year, win_num) %>%
    select(-win_num)
  
  out
}

# Generate and save summary tables
users_tbl <- make_window_table(fx_all, "users")
posts_tbl <- make_window_table(fx_all, "posts")

write_csv(users_tbl, file.path(output_folder, "p_summary_users_daily.csv"))
write_csv(posts_tbl, file.path(output_folder, "p_summary_posts_daily.csv"))

message("Summary tables written:")
message("  - ", file.path(output_folder, "p_summary_users_daily.csv"))
message("  - ", file.path(output_folder, "p_summary_posts_daily.csv"))

# ===================== VISUALIZATION (Forest plots & Event alignment, Daily only) =====================
fig_dir <- file.path(output_folder, "figs")
ensure_dir(fig_dir)

# Helper function to order windows by numeric value
# @param x: vector of window labels
# @return: ordered vector of window labels
make_window_order <- function(x) {
  tibble(window = x) %>%
    mutate(win_num = suppressWarnings(as.numeric(str_extract(window, "\\d+")))) %>%
    arrange(win_num) %>% pull(window)
}

# Create forest plots from fixed effects results
# @param fx_all: combined model results
# @param target: "users" or "posts"
make_forest_plots_from_fxall <- function(fx_all, target = c("users","posts")) {
  target <- match.arg(target)
  fx1 <- fx_all %>%
    filter(target == !!target,
           term %in% c("Pre vs Primary","Post vs Primary")) %>%
    mutate(
      term   = factor(term, levels = c("Post vs Primary","Pre vs Primary")),
      window = as.character(window)
    ) %>%
    filter(is.finite(rate_ratio), is.finite(rr_low), is.finite(rr_up))
  
  if (nrow(fx1) == 0) { 
    message("[forest] Nothing to plot for ", target)
    return(invisible(NULL)) 
  }
  
  win_order <- make_window_order(unique(fx1$window))
  fx1 <- fx1 %>% mutate(window = factor(window, levels = win_order))
  
  p <- ggplot(fx1, aes(x = rate_ratio, y = term)) +
    geom_point(size = 2.6) +
    geom_errorbarh(aes(xmin = rr_low, xmax = rr_up), height = 0.18, linewidth = 0.6) +
    geom_vline(xintercept = 1, linetype = "dashed") +
    scale_x_continuous() +
    facet_grid(year ~ window, switch = "y", scales = "free_y") +
    labs(
      title   = paste0("IRR Estimates by Daily Window — ", tools::toTitleCase(target)),
      x       = "Incidence Rate Ratio (IRR)",
      y       = NULL,
      caption = "Dot = IRR; line = 95% CI; dashed line = null effect (IRR = 1)."
    ) +
    theme_minimal(base_size = 11) +
    theme(
      panel.grid.minor = element_blank(),
      strip.placement  = "outside",
      strip.background = element_rect(fill = "grey95", color = NA),
      plot.title       = element_text(hjust = 0.5, face = "bold"),
      axis.text.x      = element_text(angle = 45, hjust = 1)
    )
  
  out <- file.path(fig_dir, paste0("p_forest_daily_", target, ".png"))
  ggsave(out, p,
         width  = max(10, 2 + 1.2 * dplyr::n_distinct(fx1$window)),
         height = max(4, 1.4 * dplyr::n_distinct(fx1$year)),
         dpi    = 300)
  message("[forest] Saved: ", out)
}

# Build daily time series for event alignment
# @param hist_csv: path to historical data CSV
# @param authors_csv: path to authors CSV
# @return: combined daily posts and users data
build_daily_series <- function(hist_csv, authors_csv) {
  hist <- readr::read_csv(hist_csv, show_col_types = FALSE) %>%
    clean_names() %>% mutate(time = as.Date(time))
  authors <- readr::read_csv(authors_csv, show_col_types = FALSE) %>%
    clean_names() %>% mutate(state = str_to_upper(str_trim(state)))
  
  dat <- hist %>% inner_join(authors[, c("author","state")], by = "author")
  
  daily_posts <- dat %>% 
    group_by(state, date = time) %>% 
    summarise(posts = sum(num, na.rm = TRUE), .groups = "drop")
  
  daily_users <- dat %>% 
    distinct(state, date = time, author) %>% 
    count(state, date, name = "users")
  
  daily_posts %>% 
    full_join(daily_users, by = c("state","date")) %>% 
    replace_na(list(posts = 0, users = 0))
}

# Create event-aligned data with fixed baseline
# @param daily: daily time series data
# @param primary_df: primary election dates
# @param target: "users" or "posts"
# @param window_days: number of days around primary
# @param base_from: baseline period start (negative days)
# @param base_to: baseline period end (negative days)
# @param min_baseline_days: minimum baseline observations
# @param rel_span: total span for relative days
# @return: aggregated event-aligned data
event_aligned_days_fixed_base <- function(daily, primary_df, target = c("users","posts"),
                                          window_days = 90, base_from = -90, base_to = -8,
                                          min_baseline_days = 10L, rel_span = 120L) {
  target <- match.arg(target)
  eps <- 1e-8
  
  have <- intersect(unique(daily$state), unique(primary_df$state))
  primary <- primary_df %>% dplyr::filter(state %in% have)
  
  if (nrow(primary) == 0) return(tibble::tibble())
  
  aligned_full <- purrr::map_dfr(seq_len(nrow(primary)), function(i) {
    st <- primary$state[i]; pday <- primary$date[i]; yy <- primary$year[i]
    g <- daily %>% dplyr::filter(state == st)
    if (nrow(g) == 0) return(tibble::tibble())
    
    g %>% 
      dplyr::mutate(rel = as.integer(difftime(date, pday, units = "days")), year = yy) %>%
      dplyr::filter(rel >= -rel_span, rel <= rel_span) %>%
      dplyr::transmute(state, year, rel, value = .data[[target]])
  })
  
  if (nrow(aligned_full) == 0) return(tibble::tibble())
  
  base_df <- aligned_full %>%
    dplyr::filter(rel >= base_from, rel <= base_to) %>%
    dplyr::group_by(state, year) %>%
    dplyr::summarise(baseline = mean(value, na.rm = TRUE),
                     n_days = sum(is.finite(value)), .groups = "drop") %>%
    dplyr::filter(is.finite(baseline), baseline > 0, n_days >= min_baseline_days)
  
  scaled <- aligned_full %>%
    dplyr::inner_join(base_df[, c("state","year","baseline")], by = c("state","year")) %>%
    dplyr::mutate(value_scaled = value / (baseline + eps))
  
  scaled %>%
    dplyr::filter(rel >= -window_days, rel <= window_days) %>%
    dplyr::group_by(year, rel) %>%
    dplyr::summarise(mean_scaled = mean(value_scaled, na.rm = TRUE),
                     n_states    = dplyr::n_distinct(state), .groups = "drop") %>%
    dplyr::mutate(window = paste0("±", window_days, "d"))
}

# Create event-aligned plots for all windows in one figure
# @param hist_csv: path to historical data
# @param authors_csv: path to authors data
# @param primary_csv: path to primary dates
# @param target: "users" or "posts"
# @param day_windows: vector of day windows to analyze
# @param save_name: custom save path (optional)
make_event_aligned_all_windows_one_figure <- function(hist_csv, authors_csv, primary_csv,
                                                      target = c("users","posts"),
                                                      day_windows = c(3,7,15,30,60,90),
                                                      base_from = -90, base_to = -8,
                                                      min_baseline_days = 10L,
                                                      rel_span = 120L,
                                                      save_name = NULL) {
  target <- match.arg(target)
  daily <- build_daily_series(hist_csv, authors_csv)
  primary_df <- readr::read_csv(primary_csv, show_col_types = FALSE) %>%
    clean_names() %>% 
    mutate(state = str_to_upper(str_trim(state)), 
           date = as.Date(date), 
           year = as.integer(year))
  
  day_list <- lapply(day_windows, function(wd)
    event_aligned_days_fixed_base(daily, primary_df, target, wd, base_from, base_to, min_baseline_days, rel_span))
  
  all_agg <- bind_rows(bind_rows(day_list))
  
  if (nrow(all_agg) == 0) { 
    message("[timeseries] Empty aggregate."); 
    return(invisible(NULL)) 
  }
  
  six_windows <- c("±3d","±7d","±15d","±30d","±60d","±90d")
  plot_df <- all_agg %>%
    mutate(window = factor(window, levels = six_windows),
           year   = factor(year))
  
  p <- ggplot(plot_df, aes(x = rel, y = mean_scaled, color = year, group = year)) +
    geom_hline(yintercept = 1, linetype = "dashed") +
    geom_vline(xintercept = 0, linetype = "dotted") +
    geom_line(linewidth = 0.9, alpha = 0.95) +
    facet_wrap(~ window, ncol = 3, scales = "free_x", drop = FALSE) +
    scale_x_continuous(name = "Relative days (0 = primary)", breaks = scales::pretty_breaks(5)) +
    labs(
      title = paste0("Scaled Reddit Activity Around Primaries (Daily) — ", tools::toTitleCase(target)),
      y = "Scaled value (mean across states)", color = "Year"
    ) +
    theme_minimal(base_size = 12) +
    theme(panel.grid.minor = element_blank(),
          plot.title = element_text(hjust = 0.5, face = "bold"),
          strip.text = element_text(face = "bold"),
          legend.position = "bottom")
  
  out_file <- if (is.null(save_name)) file.path(fig_dir, paste0("p_timeseries_daily_", target, ".png")) else save_name
  ggsave(out_file, p, width = 14, height = 8.5, dpi = 300)
  message("[timeseries] Saved: ", out_file)
}

# Create forest plot for 7-day window comparing users and posts
# @param fx_all: combined model results
make_forest_7d_both <- function(fx_all) {
  fx7 <- fx_all %>%
    dplyr::filter(window == "±7d",
                  term %in% c("Pre vs Primary","Post vs Primary"),
                  target %in% c("users","posts")) %>%
    dplyr::mutate(
      term   = factor(term, levels = c("Post vs Primary","Pre vs Primary")),
      target = factor(target, levels = c("users","posts"),
                      labels = c("Unique Users","Posts"))
    ) %>%
    dplyr::filter(is.finite(rate_ratio), is.finite(rr_low), is.finite(rr_up))
  
  if (nrow(fx7) == 0) { 
    message("[forest-7d-both] Nothing to plot."); 
    return(invisible(NULL)) 
  }
  
  p <- ggplot(fx7, aes(x = rate_ratio, y = term)) +
    geom_point(size = 3) +
    geom_errorbarh(aes(xmin = rr_low, xmax = rr_up), height = 0.22, linewidth = 0.7) +
    geom_vline(xintercept = 1, linetype = "dashed") +
    facet_grid(year ~ target, switch = "y") +
    labs(
      title   = "IRR (±7d) — Unique Users vs Posts",
      x       = "Incidence Rate Ratio (IRR)",
      y       = NULL,
      caption = "Dot = IRR; line = 95% CI; dashed = IRR = 1."
    ) +
    theme_minimal(base_size = 12) +
    theme(
      panel.grid.minor = element_blank(),
      strip.placement  = "outside",
      strip.background = element_rect(fill = "grey95", color = NA),
      plot.title       = element_text(hjust = 0.5, face = "bold")
    )
  
  out <- file.path(fig_dir, "p_forest_daily_7d.png")
  ggsave(out, p, width = 9.5, height = 4.2, dpi = 300)
  message("[forest-7d-both] Saved: ", out)
}

# Create event-aligned plot for 90-day window comparing users and posts
# @param hist_csv: path to historical data
# @param authors_csv: path to authors data
# @param primary_csv: path to primary dates
make_event_aligned_90d_both <- function(hist_csv, authors_csv, primary_csv) {
  daily <- build_daily_series(hist_csv, authors_csv)
  primary_df <- readr::read_csv(primary_csv, show_col_types = FALSE) %>%
    clean_names() %>%
    dplyr::mutate(state = str_to_upper(str_trim(state)),
                  date  = as.Date(date),
                  year  = as.integer(year))
  
  # Create aligned data for both users and posts
  df_users <- event_aligned_days_fixed_base(daily, primary_df, target = "users",
                                            window_days = 90, base_from = -90, base_to = -8,
                                            min_baseline_days = 10L, rel_span = 120L) %>%
    dplyr::mutate(target = "Unique Users")
  
  df_posts <- event_aligned_days_fixed_base(daily, primary_df, target = "posts",
                                            window_days = 90, base_from = -90, base_to = -8,
                                            min_baseline_days = 10L, rel_span = 120L) %>%
    dplyr::mutate(target = "Posts")
  
  df <- dplyr::bind_rows(df_users, df_posts)
  if (nrow(df) == 0) { 
    message("[timeseries-90d-both] Empty aggregate."); 
    return(invisible(NULL)) 
  }
  
  df <- df %>% dplyr::mutate(target = factor(target, levels = c("Unique Users", "Posts")))
  
  # X-axis breaks
  brks <- sort(unique(c(-90,-60,-30,-15,-7,-3, 0, 3,7,15,30,60,90)))
  
  # Y-axis limits
  y_max <- max(df$mean_scaled, na.rm = TRUE)
  y_lim <- c(0, max(1.05, y_max * 1.05))
  
  p <- ggplot(df, aes(x = rel, y = mean_scaled, color = factor(year), group = factor(year))) +
    geom_hline(yintercept = 1, linetype = "dashed") +
    geom_vline(xintercept = 0, linetype = "dotted") +
    geom_line(linewidth = 1) +
    facet_wrap(~ target, ncol = 1, scales = "fixed") +
    scale_x_continuous(name = "Relative days (0 = primary)", breaks = brks) +
    scale_y_continuous(limits = y_lim, expand = expansion(mult = c(0.01, 0.03))) +
    labs(
      title = "Event-Aligned (±90d) — Unique Users vs Posts",
      y     = "Scaled value (mean across states)",
      color = "Year"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      panel.grid.minor = element_blank(),
      plot.title = element_text(hjust = 0.5, face = "bold"),
      strip.text = element_text(face = "bold"),
      legend.position = "bottom"
    )
  
  out <- file.path(fig_dir, "p_timeseries_daily_90d.png")
  ggsave(out, p, width = 8.8, height = 6.0, dpi = 300)
  message("[timeseries-90d-both] Saved: ", out)
}

# ===================== GENERATE PLOTS =====================
message("Generating plots...")

try(make_forest_plots_from_fxall(fx_all, "users"), silent = TRUE)
try(make_forest_plots_from_fxall(fx_all, "posts"), silent = TRUE)
try(make_event_aligned_all_windows_one_figure(hist_csv, authors_csv, primary_csv, "users"), silent = TRUE)
try(make_event_aligned_all_windows_one_figure(hist_csv, authors_csv, primary_csv, "posts"), silent = TRUE)
try(make_forest_7d_both(fx_all), silent = TRUE)
try(make_event_aligned_90d_both(hist_csv, authors_csv, primary_csv), silent = TRUE)
message("\nAll daily-only figures written to: ", file.path(output_folder, "figs"))
