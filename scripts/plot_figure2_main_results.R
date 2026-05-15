library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(patchwork)
library(svglite)
library(ragg)
library(grid)

root_dir <- normalizePath(".", winslash = "/", mustWork = TRUE)
input_dir <- file.path(root_dir, "outputs", "figure2_inputs")
output_dir <- file.path(root_dir, "outputs", "figures_20260512_main")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

palette_contract <- c(
  neutral_dark = "#272727",
  neutral_mid = "#767676",
  neutral_light = "#D8D8D8",
  signal_blue = "#3182BD",
  signal_teal = "#33B5A5",
  accent_red = "#D24B40",
  accent_orange = "#E28E2C",
  accent_gold = "#C8A03A",
  accent_green = "#4D8B31"
)

theme_nature_contract <- function(base_size = 6.8, base_family = "Arial") {
  theme_classic(base_size = base_size, base_family = base_family) +
    theme(
      axis.line = element_line(linewidth = 0.35, colour = "black"),
      axis.ticks = element_line(linewidth = 0.35, colour = "black"),
      axis.title = element_text(size = base_size),
      axis.text = element_text(size = base_size - 0.4, colour = "black"),
      legend.title = element_text(size = base_size - 0.1),
      legend.text = element_text(size = base_size - 0.5),
      strip.text = element_text(size = base_size - 0.1, face = "bold"),
      plot.title = element_text(size = base_size + 0.5, face = "bold"),
      plot.subtitle = element_text(size = base_size - 0.3, colour = palette_contract[["neutral_dark"]]),
      plot.margin = margin(5.5, 6, 5.5, 6),
      panel.grid = element_blank()
    )
}

theme_set(theme_nature_contract())

save_pub_r <- function(plot, filename, width_mm = 183, height_mm = 215, dpi = 600) {
  w <- width_mm / 25.4
  h <- height_mm / 25.4

  svglite::svglite(paste0(filename, ".svg"), width = w, height = h)
  print(plot)
  dev.off()

  grDevices::cairo_pdf(paste0(filename, ".pdf"), width = w, height = h, family = "Arial")
  print(plot)
  dev.off()

  ragg::agg_tiff(paste0(filename, ".tiff"), width = w, height = h, units = "in", res = dpi)
  print(plot)
  dev.off()

  ragg::agg_png(paste0(filename, ".png"), width = w, height = h, units = "in", res = 220)
  print(plot)
  dev.off()
}

short_trait_label <- function(x) {
  vapply(x, function(one) {
    parts <- strsplit(one, "_", fixed = TRUE)[[1]]
    if (length(parts) %% 2 == 0) {
      half <- length(parts) / 2
      if (identical(parts[seq_len(half)], parts[(half + 1):length(parts)])) {
        return(paste(parts[seq_len(half)], collapse = "_"))
      }
    }
    one
  }, character(1))
}

dataset_pretty <- c(
  cotton1245 = "Cotton",
  rice529 = "Rice",
  soybean951 = "Soybean",
  wheat406 = "Wheat"
)

single_long <- readr::read_csv(
  file.path(input_dir, "figure2_single_long.csv"),
  show_col_types = FALSE
)

compare_sum <- readr::read_csv(
  file.path(input_dir, "figure2_compare_sum.csv"),
  show_col_types = FALSE
)

triple_trait <- readr::read_csv(
  file.path(input_dir, "figure2_triple_trait.csv"),
  show_col_types = FALSE
)

triple_vs_best <- readr::read_csv(
  file.path(input_dir, "figure2_triple_vs_best.csv"),
  show_col_types = FALSE
)

single_long <- single_long %>%
  mutate(
    dataset = factor(dataset, levels = c("cotton1245", "rice529", "soybean951", "wheat406")),
    dataset_label = recode(as.character(dataset), !!!dataset_pretty),
    prior_name = factor(prior_name, levels = c("BayesB", "GBLUP", "RKHS")),
    prior_label = factor(
      prior_name,
      levels = c("BayesB", "GBLUP", "RKHS"),
      labels = c("Single-BayesB", "Single-GBLUP", "Single-RKHS")
    ),
    trait_label = short_trait_label(trait_slug)
  )

triple_trait <- triple_trait %>%
  mutate(
    dataset = factor(dataset, levels = c("cotton1245", "rice529", "soybean951", "wheat406")),
    dataset_label = recode(as.character(dataset), !!!dataset_pretty),
    trait_label = short_trait_label(trait_slug)
  )

triple_vs_best <- triple_vs_best %>%
  mutate(
    dataset = factor(dataset, levels = c("cotton1245", "rice529", "soybean951", "wheat406")),
    dataset_label = recode(as.character(dataset), !!!dataset_pretty),
    trait_label = short_trait_label(trait_slug),
    triple_better = triple_minus_best_pct >= 0
  )

single_mean <- single_long %>%
  group_by(prior_label) %>%
  summarise(mean_gain = mean(single_vs_own_prior_pct), .groups = "drop")

panel_a <- ggplot(
  single_long,
  aes(x = prior_label, y = single_vs_own_prior_pct, fill = prior_label)
) +
  geom_hline(yintercept = 0, linetype = 2, linewidth = 0.3, colour = palette_contract[["neutral_mid"]]) +
  geom_boxplot(width = 0.62, outlier.shape = NA, linewidth = 0.3, alpha = 0.88) +
  geom_jitter(width = 0.12, height = 0, size = 0.95, alpha = 0.45, colour = palette_contract[["neutral_dark"]]) +
  geom_point(
    data = single_mean,
    aes(x = prior_label, y = mean_gain),
    inherit.aes = FALSE,
    shape = 23,
    size = 2.3,
    stroke = 0.2,
    fill = "white",
    colour = "black"
  ) +
  geom_text(
    data = single_mean,
    aes(x = prior_label, y = mean_gain, label = sprintf("%.2f%%", mean_gain)),
    inherit.aes = FALSE,
    nudge_y = 0.22,
    size = 2.2
  ) +
  scale_fill_manual(
    values = c(
      "Single-BayesB" = palette_contract[["signal_blue"]],
      "Single-GBLUP" = palette_contract[["signal_teal"]],
      "Single-RKHS" = palette_contract[["accent_orange"]]
    )
  ) +
  labs(
    x = NULL,
    y = "Gain over matched prior (%)",
    title = "Single-prior fusion improves matched priors"
  ) +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 18, hjust = 1)
  )

triple_long <- triple_trait %>%
  transmute(
    dataset,
    dataset_label,
    trait_label,
    comparator = "BayesB",
    gain_pct = triple_vs_BayesB_pct
  ) %>%
  bind_rows(
    triple_trait %>%
      transmute(dataset, dataset_label, trait_label, comparator = "GBLUP", gain_pct = triple_vs_GBLUP_pct),
    triple_trait %>%
      transmute(dataset, dataset_label, trait_label, comparator = "RKHS", gain_pct = triple_vs_RKHS_pct),
    triple_trait %>%
      transmute(dataset, dataset_label, trait_label, comparator = "Best baseline", gain_pct = triple_vs_best_baseline_pct),
    triple_trait %>%
      transmute(dataset, dataset_label, trait_label, comparator = "No-prior TabICL", gain_pct = triple_vs_no_prior_tabicl_pct)
  ) %>%
  mutate(
    comparator = factor(
      comparator,
      levels = c("BayesB", "GBLUP", "RKHS", "Best baseline", "No-prior TabICL")
    )
  )

triple_mean <- triple_long %>%
  group_by(comparator) %>%
  summarise(mean_gain = mean(gain_pct), .groups = "drop")

panel_b_ylim_upper <- 12
panel_b_outliers <- triple_long %>%
  filter(gain_pct > panel_b_ylim_upper) %>%
  mutate(plot_y = panel_b_ylim_upper - 0.35)

panel_b <- ggplot(
  triple_long,
  aes(x = comparator, y = gain_pct, fill = comparator)
) +
  geom_hline(yintercept = 0, linetype = 2, linewidth = 0.3, colour = palette_contract[["neutral_mid"]]) +
  geom_boxplot(width = 0.63, outlier.shape = NA, linewidth = 0.3, alpha = 0.88) +
  geom_jitter(width = 0.12, height = 0, size = 0.95, alpha = 0.45, colour = palette_contract[["neutral_dark"]]) +
  geom_point(
    data = triple_mean,
    aes(x = comparator, y = mean_gain),
    inherit.aes = FALSE,
    shape = 23,
    size = 2.3,
    stroke = 0.2,
    fill = "white",
    colour = "black"
  ) +
  geom_text(
    data = triple_mean,
    aes(x = comparator, y = mean_gain, label = sprintf("%.2f%%", mean_gain)),
    inherit.aes = FALSE,
    nudge_y = 0.28,
    size = 2.2
  ) +
  geom_point(
    data = panel_b_outliers,
    aes(x = comparator, y = plot_y),
    inherit.aes = FALSE,
    shape = 24,
    size = 2.1,
    stroke = 0.25,
    fill = palette_contract[["accent_red"]],
    colour = "black"
  ) +
  annotate(
    "text",
    x = 4.55,
    y = panel_b_ylim_upper - 0.05,
    label = sprintf("%d traits clipped above %d%%", nrow(panel_b_outliers), panel_b_ylim_upper),
    size = 1.95,
    vjust = 0,
    colour = palette_contract[["neutral_dark"]]
  ) +
  scale_fill_manual(
    values = c(
      "BayesB" = palette_contract[["signal_blue"]],
      "GBLUP" = palette_contract[["signal_teal"]],
      "RKHS" = palette_contract[["accent_orange"]],
      "Best baseline" = palette_contract[["accent_gold"]],
      "No-prior TabICL" = palette_contract[["accent_red"]]
    )
  ) +
  labs(
    x = NULL,
    y = "Gain of triple-prior fusion (%)",
    title = "Triple-prior fusion improves over baselines\nand no-prior TabICL"
  ) +
  coord_cartesian(ylim = c(-1.5, panel_b_ylim_upper), clip = "on") +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 18, hjust = 1)
  )

heat_single <- single_long %>%
  select(dataset, dataset_label, trait_label, prior_label, single_vs_own_prior_pct) %>%
  mutate(method = as.character(prior_label), gain_pct = single_vs_own_prior_pct) %>%
  select(dataset, dataset_label, trait_label, method, gain_pct)

heat_triple <- triple_trait %>%
  transmute(
    dataset,
    dataset_label,
    trait_label,
    method = "Triple (vs only-triple)",
    gain_pct = triple_vs_only_triple_pct
  )

heat_df <- bind_rows(heat_single, heat_triple) %>%
  mutate(
    method = factor(
      method,
      levels = c("Single-BayesB", "Single-GBLUP", "Single-RKHS", "Triple (vs only-triple)")
    )
  )

trait_order <- triple_trait %>%
  arrange(dataset, desc(triple_vs_only_triple_pct), trait_label) %>%
  mutate(trait_id = paste(as.character(dataset), trait_label, sep = "|||")) %>%
  pull(trait_id)

heat_df <- heat_df %>%
  mutate(
    trait_id = factor(
      paste(as.character(dataset), trait_label, sep = "|||"),
      levels = rev(unique(trait_order))
    )
  )

panel_c <- ggplot(heat_df, aes(x = method, y = trait_id, fill = gain_pct)) +
  geom_tile(colour = "white", linewidth = 0.25) +
  facet_grid(dataset_label ~ ., scales = "free_y", space = "free_y", switch = "y") +
  scale_y_discrete(labels = function(x) sub("^.*\\|\\|\\|", "", x)) +
  scale_fill_gradient2(
    low = palette_contract[["accent_red"]],
    mid = "#FFF9F0",
    high = palette_contract[["signal_blue"]],
    midpoint = 0,
    name = "Gain (%)"
  ) +
  labs(
    x = NULL,
    y = NULL,
    title = "Trait-level gains remain heterogeneous",
    subtitle = "Single columns are measured against the matched prior-only model; triple is measured against only-triple-prior"
  ) +
  theme(
    axis.text.x = element_text(angle = 22, hjust = 1, vjust = 1),
    axis.text.y = element_text(size = 4.7),
    axis.ticks = element_blank(),
    strip.background = element_rect(fill = "#F2F2F2", colour = NA),
    strip.placement = "outside",
    strip.text.y.left = element_text(angle = 0, hjust = 1),
    panel.spacing.y = unit(0.25, "lines"),
    legend.position = "right"
  )

dumbbell_df <- triple_vs_best %>%
  arrange(dataset, desc(triple_minus_best_pct), trait_label) %>%
  mutate(
    trait_id = factor(
      paste(as.character(dataset), trait_label, sep = "|||"),
      levels = rev(unique(paste(as.character(dataset), trait_label, sep = "|||")))
    ),
    direction = ifelse(triple_better, ">=0%", "<0%")
  )

panel_d <- ggplot(dumbbell_df) +
  geom_segment(
    aes(
      x = best_baseline,
      xend = triple_two_step_ls,
      y = trait_id,
      yend = trait_id,
      colour = direction
    ),
    linewidth = 0.55,
    alpha = 0.85
  ) +
  geom_point(
    aes(x = best_baseline, y = trait_id),
    shape = 21,
    size = 1.65,
    stroke = 0.25,
    fill = "white",
    colour = palette_contract[["neutral_dark"]]
  ) +
  geom_point(
    aes(x = triple_two_step_ls, y = trait_id, fill = direction),
    shape = 21,
    size = 1.85,
    stroke = 0.25,
    colour = "black"
  ) +
  facet_grid(dataset_label ~ ., scales = "free_y", space = "free_y", switch = "y") +
  scale_y_discrete(labels = function(x) sub("^.*\\|\\|\\|", "", x)) +
  scale_colour_manual(
    values = c(
      ">=0%" = palette_contract[["accent_green"]],
      "<0%" = palette_contract[["neutral_light"]]
    )
  ) +
  scale_fill_manual(
    values = c(
      ">=0%" = palette_contract[["accent_green"]],
      "<0%" = palette_contract[["neutral_light"]]
    )
  ) +
  labs(
    x = "Pearson correlation",
    y = NULL,
    title = sprintf(
      "Triple usually beats the best baseline (%d/%d non-negative)",
      sum(dumbbell_df$triple_minus_best_pct >= 0),
      nrow(dumbbell_df)
    )
  ) +
  guides(
    colour = guide_legend(nrow = 1, byrow = TRUE),
    fill = "none"
  ) +
  theme(
    axis.text.y = element_text(size = 4.7),
    axis.ticks.y = element_blank(),
    strip.background = element_rect(fill = "#F2F2F2", colour = NA),
    strip.placement = "outside",
    strip.text.y.left = element_text(angle = 0, hjust = 1),
    panel.spacing.y = unit(0.25, "lines"),
    legend.position = "top",
    legend.title = element_blank(),
    legend.text = element_text(size = 5.4),
    plot.title = element_text(size = 6.4, face = "bold", lineheight = 1.03)
  )

figure_2 <- ((panel_a | panel_b) / (panel_c | panel_d)) +
  plot_annotation(
    title = "Prior-integrated fusion consistently strengthens TabICL in genomic prediction",
    subtitle = "Main results across 36 non-pig traits under the 5.4-duli-liudang pipeline",
    tag_levels = "a"
  ) &
  theme(
    plot.tag = element_text(size = 8, face = "bold"),
    plot.title = element_text(size = 7.7, face = "bold", lineheight = 1.05),
    plot.subtitle = element_text(size = 6.6)
  )

outfile <- file.path(output_dir, "figure2_main_results")
save_pub_r(figure_2, outfile, width_mm = 183, height_mm = 220)

message("Saved Figure 2 to: ", outfile)
