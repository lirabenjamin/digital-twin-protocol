# Step 9. Test each candidate moderator of the effect with a mixed model, one moderator at a time.
#
#   Rscript scripts/moderation.R output/choice_rejection   # after run_twins.py and rate_moderators.py
#   Rscript scripts/moderation.R output/fake                # the synthetic data from make_fake_data.py
#
# Reads <dir>/choices.csv (pid, stimulus, treat, y) and <dir>/stimulus_moderators.csv (stimulus + one
# column per moderator, the mean of the 0-10 ratings).
#
# Reference model:  y ~ treat + (1 | pid) + (1 + treat | stimulus)
# Per moderator:    y ~ treat * z + (1 | pid) + (1 + treat | stimulus)
# z is the moderator standardized across stimuli. The treat:z coefficient (b) is the change in the effect per
# SD of the moderator, in DV units. rel = b as a share of the effect at the average stimulus. het_explained =
# the drop in the variance of the stimulus random slope relative to the reference model. p values are Wald;
# bh = Benjamini-Hochberg across all moderators. r_stimulus = plain correlation between the moderator and the
# per-stimulus effect. If y is 0/1 the models are logistic (glmer) and b is on the log-odds scale.
#
# Writes <dir>/moderation.csv, moderator_cor.csv (moderator x moderator correlations across stimuli),
# moderator_pca.csv and moderator_cor.png.
suppressMessages(library(lme4))

args <- commandArgs(trailingOnly = TRUE)
dir <- if (length(args)) args[1] else "output/choice_rejection"
d <- read.csv(file.path(dir, "choices.csv"))
d <- d[!is.na(d$y), c("pid", "stimulus", "treat", "y")]
d$pid <- factor(d$pid)
raw <- read.csv(file.path(dir, "stimulus_moderators.csv"), check.names = FALSE)
mods <- setdiff(names(raw), "stimulus")
z <- raw
z[mods] <- lapply(raw[mods], function(x) as.numeric(scale(x)))

binary <- all(d$y %in% c(0, 1))
fit <- function(f, data) {
  if (binary) glmer(f, data = data, family = binomial, control = glmerControl(calc.derivs = FALSE))
  else lmer(f, data = data, REML = FALSE, control = lmerControl(calc.derivs = FALSE))
}
slope_var <- function(m) { v <- as.data.frame(VarCorr(m)); v$vcov[v$grp == "stimulus" & v$var1 == "treat" & is.na(v$var2)] }
re <- "(1 | pid) + (1 + treat | stimulus)"
ref <- fit(as.formula(paste("y ~ treat +", re)), d)

eff <- aggregate(y ~ stimulus + treat, d, mean)
eff <- merge(eff[eff$treat == 1, c("stimulus", "y")], eff[eff$treat == 0, c("stimulus", "y")], by = "stimulus")
eff$effect <- eff$y.x - eff$y.y

rows <- lapply(mods, function(m) {
  if (isTRUE(sd(raw[[m]], na.rm = TRUE) == 0)) return(data.frame(moderator = m, note = "no variance"))
  dz <- merge(d, data.frame(stimulus = z$stimulus, z = z[[m]]), by = "stimulus")
  fm <- fit(y ~ treat * z + (1 | pid) + (1 + treat | stimulus), dz)
  b <- fixef(fm)[["treat:z"]]; se <- sqrt(vcov(fm)["treat:z", "treat:z"])
  data.frame(moderator = m, b = b, se = se, p = 2 * pnorm(-abs(b / se)), effect_at_mean = fixef(fm)[["treat"]],
             rel = b / fixef(fm)[["treat"]], het_explained = 1 - slope_var(fm) / slope_var(ref),
             r_stimulus = cor(raw[[m]][match(eff$stimulus, raw$stimulus)], eff$effect, use = "complete.obs"),
             converged = length(fm@optinfo$conv$lme4$messages) == 0, note = "")
})
all_cols <- unique(unlist(lapply(rows, names)))
res <- do.call(rbind, lapply(rows, function(r) { r[setdiff(all_cols, names(r))] <- NA; r[all_cols] }))
res$bh <- p.adjust(res$p, "BH")
res <- res[order(res$p), ]
write.csv(res, file.path(dir, "moderation.csv"), row.names = FALSE)

# Moderators that measure the same thing will both look significant. Their correlations and a PCA show how
# many separate dimensions the catalog holds.
ok <- mods[sapply(raw[mods], function(x) isTRUE(sd(x, na.rm = TRUE) > 0))]
cm <- cor(raw[ok], use = "pairwise.complete.obs")
write.csv(cm, file.path(dir, "moderator_cor.csv"))
pca <- prcomp(na.omit(raw[ok]), scale. = TRUE)
pve <- pca$sdev^2 / sum(pca$sdev^2)
write.csv(data.frame(pc = seq_along(pve), share = pve, cumulative = cumsum(pve)),
          file.path(dir, "moderator_pca.csv"), row.names = FALSE)
ord <- hclust(as.dist(1 - cm))$order
png(file.path(dir, "moderator_cor.png"), width = 1400, height = 1300, res = 150)
par(mar = c(9, 9, 3, 1))
image(seq_along(ok), seq_along(ok), cm[ord, ord], zlim = c(-1, 1), axes = FALSE, xlab = "", ylab = "",
      col = colorRampPalette(c("#2166ac", "white", "#b2182b"))(41),
      main = "Correlations among the moderators (across stimuli)")
axis(1, seq_along(ok), ok[ord], las = 2, cex.axis = 0.6)
axis(2, seq_along(ok), ok[ord], las = 2, cex.axis = 0.6)
invisible(dev.off())

cat(sprintf("%s models; effect at the average stimulus %.3f (reference model)\n",
            if (binary) "logistic" else "linear", fixef(ref)[["treat"]]))
cat(sprintf("BH < .05: %d of %d moderators; moderator pairs with |r| > .7: %d of %d; components for 80%% of variance: %d\n",
            sum(res$bh < .05, na.rm = TRUE), nrow(res), sum(abs(cm[upper.tri(cm)]) > .7), sum(upper.tri(cm)),
            which(cumsum(pve) >= .8)[1]))
num <- sapply(res, is.numeric)
res[num] <- lapply(res[num], signif, 3)
print(res[, c("moderator", "b", "se", "p", "bh", "rel", "het_explained", "r_stimulus", "converged")], row.names = FALSE)
