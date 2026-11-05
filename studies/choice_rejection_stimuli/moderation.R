# Moderation of the rejection-frame effect by each of the 100 candidate moderators, one at a time, as for
# De La Rosa (studies/delarosa_stimuli/moderation_mlm.R), with a linear model because the DV is the 7-point pref_b.
#
#   Rscript moderation.R        # fits are cached in output/moderators/fits/; delete it to refit
#
# Reference: pref_b ~ rejection + (1 | pid) + (1 + rejection | stimulus)
# Per moderator: pref_b ~ rejection * z + (1 | pid) + (1 + rejection | stimulus)
# z is the moderator's mean rating, standardized across the 100 stimuli. The rejection:z coefficient is the
# moderation: the change in the frame effect (points on the 7-point scale) per SD of the moderator. "rel" is that
# change as a share of the frame effect at the average stimulus. "het_explained" is the drop in the variance of
# the stimulus frame slope relative to the reference model. p values are Wald (normal); BH across the 100.
# Also writes the 100 x 100 correlation matrix of the moderators and a principal-components summary.
# Output: output/moderators/moderation.csv, moderator_cor.csv, moderator_pca.csv, moderator_cor.png
suppressMessages({library(tidyverse); library(lme4); library(parallel)})

out <- "output/moderators"
cache <- file.path(out, "fits"); dir.create(cache, showWarnings = FALSE, recursive = TRUE)
raw <- read_csv(file.path(out, "stimulus_moderators.csv"), show_col_types = FALSE)
mods <- setdiff(names(raw), "stimulus")
stim <- raw |> mutate(across(all_of(mods), \(x) as.numeric(scale(x))))
d <- read_csv("output/full/choices.csv", show_col_types = FALSE) |> filter(!is.na(pref_b)) |>
  transmute(pid = factor(pid), stimulus, rejection = as.integer(frame == "rejection"), pref_b) |>
  left_join(stim, by = "stimulus")

ctrl <- lmerControl(calc.derivs = FALSE)
slope_var <- function(m) { v <- as.data.frame(VarCorr(m)); v$vcov[v$grp == "stimulus" & v$var1 == "rejection" & is.na(v$var2)] }
cached_fit <- function(name, f, data) {
  path <- file.path(cache, paste0(name, ".rds"))
  if (file.exists(path)) return(readRDS(path))
  m <- lmer(as.formula(f), data = data, REML = FALSE, control = ctrl)
  saveRDS(m, path); m
}
re <- "(1 | pid) + (1 + rejection | stimulus)"
ref <- cached_fit("reference", paste("pref_b ~ rejection +", re), d)

rows <- mclapply(mods, \(fid) {
  dz <- d[, c("pid", "stimulus", "rejection", "pref_b")]; dz$z <- d[[fid]]
  if (sd(stim[[fid]], na.rm = TRUE) == 0 || all(is.na(stim[[fid]]))) return(tibble(moderator = fid, note = "no variance"))
  m <- cached_fit(fid, paste("pref_b ~ rejection * z +", re), dz)
  b <- fixef(m)[["rejection:z"]]; se <- sqrt(vcov(m)["rejection:z", "rejection:z"])
  k <- c("rejection", "rejection:z"); bb <- fixef(m)[k]; V <- as.matrix(vcov(m)[k, k]); g <- c(-bb[2] / bb[1]^2, 1 / bb[1])
  tibble(moderator = fid, b = b, se = se, p = 2 * pnorm(-abs(b / se)), effect_at_mean = bb[[1]],
         rel = bb[[2]] / bb[[1]], rel_se = sqrt(drop(t(g) %*% V %*% g)),
         het_explained = 1 - slope_var(m) / slope_var(ref),
         converged = length(m@optinfo$conv$lme4$messages) == 0)
}, mc.cores = max(1, min(6, detectCores() - 2)))
failed <- keep(rows, \(r) inherits(r, "try-error"))
if (length(failed)) stop(paste(map_chr(failed, as.character), collapse = "\n"))

catalog <- jsonlite::fromJSON("moderators.json") |> select(moderator = id, group, name)
# stimulus-level check: correlation of each moderator with the per-stimulus frame effect (100 points)
eff <- d |> group_by(stimulus, rejection) |> summarise(m = mean(pref_b), .groups = "drop") |>
  pivot_wider(names_from = rejection, values_from = m) |> transmute(stimulus, effect = `1` - `0`)
r_stim <- map_dbl(mods, \(fid) cor(stim[[fid]], eff$effect[match(stim$stimulus, eff$stimulus)], use = "complete.obs"))
res <- bind_rows(rows) |> mutate(bh = p.adjust(p, "BH"), r_stimulus = r_stim[match(moderator, mods)]) |>
  left_join(catalog, by = "moderator") |> relocate(moderator, group, name) |> arrange(p)
write_csv(res, file.path(out, "moderation.csv"))

# correlations among the moderators (across the 100 stimuli)
cm <- cor(stim[mods], use = "pairwise.complete.obs")
write_csv(as_tibble(cm, rownames = "moderator"), file.path(out, "moderator_cor.csv"))
ok <- mods[apply(stim[mods], 2, \(x) sd(x, na.rm = TRUE) > 0)]
pca <- prcomp(stim[ok], scale. = TRUE)
pve <- pca$sdev^2 / sum(pca$sdev^2)
write_csv(tibble(pc = seq_along(pve), eigenvalue = pca$sdev^2, share = pve, cumulative = cumsum(pve)),
          file.path(out, "moderator_pca.csv"))
ord <- hclust(as.dist(1 - cm[ok, ok]))$order
long <- as_tibble(cm[ok, ok][ord, ord], rownames = "a") |> pivot_longer(-a, names_to = "b", values_to = "r") |>
  mutate(a = factor(a, levels = ok[ord]), b = factor(b, levels = rev(ok[ord])))
p <- ggplot(long, aes(a, b, fill = r)) + geom_tile() +
  scale_fill_gradient2(low = "#2166ac", mid = "white", high = "#b2182b", limits = c(-1, 1)) +
  labs(x = NULL, y = NULL, fill = "r", title = "Correlations among the 100 candidate moderators (100 stimuli)",
       subtitle = "Ordered by hierarchical clustering on 1 - r") +
  theme_minimal(base_size = 7) + theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
ggsave(file.path(out, "moderator_cor.png"), p, width = 13, height = 12, dpi = 150, bg = "white")

offdiag <- cm[upper.tri(cm)]
cat(sprintf("moderator pairs with |r| > .7: %d of %d; components for 80%% of variance: %d\n",
            sum(abs(offdiag) > .7, na.rm = TRUE), length(offdiag), which(cumsum(pve) >= .8)[1]))
cat(sprintf("BH < .05: %d of %d moderators\n", sum(res$bh < .05, na.rm = TRUE), nrow(res)))
print(res |> select(name, b, se, p, bh, rel, het_explained, r_stimulus) |> mutate(across(where(is.numeric), \(x) signif(x, 3))), n = 20)
