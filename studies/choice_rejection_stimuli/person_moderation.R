# Moderation of the rejection-frame effect by 10 person-level traits of the twins, one trait at a time.
#
#   python3 person_moderators.py && Rscript person_moderation.R   # fits are cached in output/person/fits/
#
# Reference: pref_b ~ rejection + (1 + rejection | pid) + (1 + rejection | stimulus)
# Per trait: pref_b ~ rejection * z + (1 + rejection | pid) + (1 + rejection | stimulus)
# z is the trait standardized across the 2,058 twins. The rejection:z coefficient is the change in the frame effect
# (points on the 7-point pref_b scale) per SD of the trait; positive = twins higher on the trait move more toward B
# under the rejection frame. "het_explained" is the drop in the variance of the twin frame slope relative to the
# reference model. p values are Wald (normal); BH across the traits. Twins with a missing trait are dropped from
# that trait's model, and the reference is refit on the same twins so het_explained compares like with like.
# r_twin is the plain correlation between the trait and each twin's mean frame effect (rejection - choice).
# If the reference model with a twin random slope fails to converge, all models fall back to (1 | pid).
# Output: output/person/person_moderation.csv, trait_cor.csv
suppressMessages({library(tidyverse); library(lme4); library(parallel)})

out <- "output/person"
cache <- file.path(out, "fits"); dir.create(cache, showWarnings = FALSE, recursive = TRUE)
traits <- read_csv(file.path(out, "traits.csv"), show_col_types = FALSE)
tr <- setdiff(names(traits), "pid")
z <- traits |> mutate(across(all_of(tr), \(x) as.numeric(scale(x))))
d <- read_csv("output/full/choices.csv", show_col_types = FALSE) |> filter(!is.na(pref_b)) |>
  transmute(pid, stimulus, rejection = as.integer(frame == "rejection"), pref_b) |>
  left_join(z, by = "pid") |> mutate(pid = factor(pid))

ctrl <- lmerControl(calc.derivs = FALSE)
converged <- function(m) length(m@optinfo$conv$lme4$messages) == 0 && !isSingular(m)
slope_var <- function(m) { v <- as.data.frame(VarCorr(m)); v$vcov[v$grp == "pid" & v$var1 == "rejection" & is.na(v$var2)] }
cached_fit <- function(name, f, data) {
  path <- file.path(cache, paste0(name, ".rds"))
  if (file.exists(path)) return(readRDS(path))
  m <- lmer(as.formula(f), data = data, REML = FALSE, control = ctrl)
  saveRDS(m, path); m
}
re <- "(1 + rejection | pid) + (1 + rejection | stimulus)"
t0 <- Sys.time()
ref <- cached_fit("reference", paste("pref_b ~ rejection +", re), d)
cat(sprintf("reference fit: %.1f min, converged %s\n", as.numeric(Sys.time() - t0, units = "mins"), converged(ref)))
if (!converged(ref)) {
  re <- "(1 | pid) + (1 + rejection | stimulus)"
  cat("fallback: twin random slope dropped, using", re, "\n")
  ref <- cached_fit("reference_nopidslope", paste("pref_b ~ rejection +", re), d)
}
pid_slope <- grepl("rejection \\| pid", re)

rows <- mclapply(tr, \(t) {
  dz <- d[!is.na(d[[t]]), c("pid", "stimulus", "rejection", "pref_b", t)] |> rename(z = all_of(t))
  tag <- if (pid_slope) "" else "_nopidslope"
  m <- cached_fit(paste0(t, tag), paste("pref_b ~ rejection * z +", re), dz)
  r0 <- if (nrow(dz) == nrow(d)) ref else cached_fit(paste0("reference_", t, tag), paste("pref_b ~ rejection +", re), dz)
  b <- fixef(m)[["rejection:z"]]; se <- sqrt(vcov(m)["rejection:z", "rejection:z"])
  tibble(trait = t, n_twins = n_distinct(dz$pid), b = b, se = se, p = 2 * pnorm(-abs(b / se)),
         effect_at_mean = fixef(m)[["rejection"]], rel = b / fixef(m)[["rejection"]],
         het_explained = if (pid_slope) 1 - slope_var(m) / slope_var(r0) else NA_real_,
         converged = converged(m))
}, mc.cores = 3)
failed <- keep(rows, \(r) inherits(r, "try-error"))
if (length(failed)) stop(paste(map_chr(failed, as.character), collapse = "\n"))

# twin-level check: correlation of each trait with the twin's mean frame effect (n = 2,058)
eff <- d |> group_by(pid, rejection) |> summarise(m = mean(pref_b), .groups = "drop") |>
  pivot_wider(names_from = rejection, values_from = m) |> transmute(pid = as.integer(as.character(pid)), effect = `1` - `0`) |>
  left_join(traits, by = "pid")
ct <- map(tr, \(t) cor.test(eff[[t]], eff$effect))
res <- bind_rows(rows) |>
  mutate(bh = p.adjust(p, "BH"), r_twin = map_dbl(ct, "estimate")[match(trait, tr)],
         r_twin_p = map_dbl(ct, "p.value")[match(trait, tr)], random_effects = re) |>
  arrange(p)
write_csv(res, file.path(out, "person_moderation.csv"))
write_csv(as_tibble(cor(traits[tr], use = "pairwise.complete.obs"), rownames = "trait"), file.path(out, "trait_cor.csv"))

cat(sprintf("twin frame-slope SD in the reference model: %.3f; total run %.1f min\n",
            if (pid_slope) sqrt(slope_var(ref)) else NA, as.numeric(Sys.time() - t0, units = "mins")))
print(res |> select(trait, n_twins, b, se, p, bh, het_explained, r_twin, r_twin_p, converged) |>
        mutate(across(where(is.double), \(x) signif(x, 3))), n = Inf, width = Inf)
