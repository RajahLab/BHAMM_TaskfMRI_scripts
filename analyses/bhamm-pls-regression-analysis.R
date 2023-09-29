# Name: bhamm-pls-regression-analysis.R
# Author: Rikki Lissaman
# Last Updated: September 28, 2023
#
# Description: This R script contains the code used to analyze behavioral data for 
# "The interactive influence of chronological age and menopause status on the functional
# neural correlates of spatial context memory in middle-aged females" by Crestol
# et al. (2023).
# 
# Inputs: 
# (1) 1 SPSS file (.sav) containing relevant data from the Brain Health at Midlife 
# and Menopause (BHAMM) study (BHAM_N72_MA_31Pre41Post_MRIsample_CorrectedFA_RT.sav).
#
# Outputs:
# (1) Figure (.svg) showing correlation between independent/dependent variables 
# and latent variables from the PLS regression analysis. 
# (2) Figure (.svg) showing relationship between spatial context retrieval accuracy
# and age as a function of task in pre-menopausal females.
# (3) Figure (.svg) showing relationship between spatial context retrieval accuracy
# and age as a function of task in post-menopausal females.
#
# Additional Information:
# - Run using R version 4.1.3 (2022-03-10) on 2021 MacBook Pro (M1 chip).
# - The plot generated and exported was amended outside of R (using Inkscape) to 
# improve readability for publication. 


# LOAD PACKAGES -----------------------------------------------------------

# Load packages for data wrangling/analysis/visualization.
library(haven) # version 2.5.0
library(janitor) # version 2.1.0
library(tidyr) # version 1.2.0
library(dplyr) # version 1.0.8
library(stringr) # version 1.4.0
library(plsdepot) # version 0.2.0
library(ggplot2) # version 3.3.6
library(svglite) # version 2.1.1
library(lme4) # version 1.1-29
library(lmerTest) # version 3.1-3
library(ggeffects) # version 1.1.2

# IMPORT AND ORGANIZE THE DATA --------------------------------------------

# Import the final data set (i.e., after exclusions applied), "clean" the variable
# names (i.e., make names unique, convert all to lower-case, etc), and change the
# menopause status variable to a factor.
bhamm <- read_sav("BHAM_N72_MA_31Pre41Post_MRIsample_CorrectedFA_RT.sav") %>% 
  clean_names() %>%
  mutate(s2_meno_group = as_factor(s2_meno_group))

# Amend the menopause status variable so it contains numbers (-1, 1) instead of 
# labels ("Premenopause", "Postmenopause"), as `plsreg2` (see below) requires 
# variables be in numeric format.
bhamm <- bhamm %>% 
  mutate(s2_meno_group = case_when(s2_meno_group == "Premenopause" ~ -1,
                                   s2_meno_group == "Postmenopause" ~ 1))

# Create a new variable that contains standardized age values (i.e., z-scores). 
# Ensure it is stored as a numeric variable.
bhamm <- bhamm %>% 
  mutate(s2_age_std = scale(s2_age, center = TRUE, scale = TRUE)) %>%
  mutate(s2_age_std = as.numeric(s2_age_std))


# RUN PLS REGRESSION FOR ACCURACY DATA ------------------------------------

# Create a new tibble, x_acc, that contains the independent variables (a.k.a., the 
# predictor variables) for the accuracy analysis. In this case, the independent 
# variables are menopause status at session 2 (`s2_meno_group`) and age (years) 
# at session 2 (`s2_age_std`). 
x_acc <- bhamm %>% select(s2_meno_group, s2_age_std)

# Create a new tibble, y_acc, that contains the dependent variables (a.k.a., the outcome 
# variables) for the accuracy analysis. In this case, the dependent variables are 
# correct source rate - easy (`cs_rate_easy`), correct source rate - hard (`cs_rate_hard`), 
# recognition rate - easy (`recog_rate_easy`), and recognition rate - hard (`recog_rate_hard`). 
y_acc <- bhamm %>% select(cs_rate_easy, cs_rate_hard, recog_rate_easy, recog_rate_hard)

# Standardize (i.e., z-score) all of the dependent variables. Ensure all are stored 
# as numeric variables.
y_acc <- y_acc %>% 
  mutate_if(is.numeric, ~scale(., center = TRUE, scale = TRUE)) %>% 
  mutate_if(is.numeric, ~as.numeric(.))

# Run PLS regression analysis on the accuracy data using `plsreg2`. This performs 
# a PLS2 analysis; essentially, a PLS regression in the multivariate (or multi-response) case.
set.seed(123)
plsreg2_acc <- plsreg2(predictors = x_acc, responses = y_acc, comps = NULL, crosval = TRUE) 

# Examine the Q2 values for the LVs (a.k.a., components). Values >= 0.0975 are 
# usually considered "significant".
plsreg2_acc$Q2

# Examine the percentage of variance in X (age, menopause) and Y (performance measures) 
# explained by the LVs.
plsreg2_acc$expvar

# Examine correlations used in circle correlation plot.
plsreg2_acc$cor.xt
plsreg2_acc$cor.yt

# Generate a data frame that contains data from the PLS regression analysis.
pls_acc_corrs <- data.frame(var_name = c("Menopause", "Age", 
                                         "CS Easy", "CS Hard",
                                         "Recog Easy", "Recog Hard"),
                            var_type = c("IV", "IV", "DV", "DV", "DV", "DV"),
                            corr = c(-0.9659003, -0.9779276, 0.5093213, 0.4860441,
                                     -0.3671049, -0.2862653))

# Amend the data frame so that variables are in their correct form.
pls_acc_corrs <- pls_acc_corrs %>%
  mutate(var_name = factor(var_name, levels = c("Menopause", "Age", 
                                                "CS Easy", "CS Hard",
                                                "Recog Easy", "Recog Hard")),
         var_type = factor(var_type, levels = c("IV", "DV")))

# Generate a barplot based on the PLS regression analysis output.
pls_acc_corrs_fig <- ggplot(pls_acc_corrs, 
                            aes(x = var_name, y = corr, fill = var_type)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  theme_bw() +
  theme(axis.text.x = element_text(family = "Arial", colour = "black", size = 12),
        axis.text.y = element_text(family = "Arial", colour = "black", size = 12),
        axis.title.x = element_blank(),
        axis.title.y = element_text(family = "Arial", size = 12),
        strip.text.x = element_text(family = "Arial", colour = "black", size = 12),
        legend.text = element_text(family = "Arial", colour = "black", size = 12),
        legend.title = element_blank(),
        legend.position = "None") +
  scale_x_discrete(name = "Variables") +
  scale_y_continuous(name = "Behavior-LV Correlations", limits = c(-1, 1),
                     breaks = seq(from = -1, to = 1, by = 0.2),
                     labels = c(seq(from = -1, to = 1, by = 0.2))) +
  scale_colour_manual(values = c("#0072B2", "#D55E00")) +
  scale_fill_manual(values = c("#0072B2", "#D55E00"))
                                
# Save the barplot in svg format, which we can edit in Inkscape.
ggsave("PLSr_barplot_sept2023.svg", pls_acc_corrs_fig, width = 8, height = 4)

# Generate a "long" data set for post-hoc analyses with linear mixed-effects models.
bhamm_long <- bhamm %>%
  select(id, s2_age, s2_age_std, s2_meno_group, cs_rate_easy, cs_rate_hard) %>%
  pivot_longer(!c(id, s2_age, s2_age_std, s2_meno_group), 
               names_to = "task", values_to = "cs_rate") %>%
  mutate(id = factor(id),
         s2_meno_group = factor(s2_meno_group))

# Remove unnecessary text in the new variable ("task").
bhamm_long$task <- str_replace_all(bhamm_long$task, "cs_rate_", "")

# Convert task variable to a factor.
bhamm_long <- mutate(bhamm_long, task = factor(task, levels = c("easy","hard")))

# Generate separate data sets for pre- and post-menopausal females.
pre_long <- filter(bhamm_long, s2_meno_group == -1)
post_long <- filter(bhamm_long, s2_meno_group == 1)

# Change coding of task variables to sum in the respectie data sets.
contrasts(pre_long$task) <- contr.sum(2)
contrasts(post_long$task) <- contr.sum(2)

# Conduct linear mixed-effects analysis for pre-menopausal females.
lmm_pre <- lmerTest::lmer(cs_rate ~ task * s2_age_std + (1|id), data = pre_long)

# Generate outputs for the pre-menopausal females.
anova(lmm_pre)

# Generate predictions for the pre-menopausal model and store.
pred_pre <- ggpredict(lmm_pre, terms = c( "s2_age_std", "task"))

# Amend the predictions data set for later plotting.
pred_pre <- pred_pre %>% 
  rename(s2_age_std = x, cs_rate = predicted, task = group)

# Plot a figure based on the predicted values for pre-menopausal females.
set.seed(123) 
pred_pre_fig <- pred_pre %>%
  ggplot(aes(x = s2_age_std, y = cs_rate, colour = task)) +
  geom_line(aes(x = s2_age_std, y = cs_rate, colour = task)) +
  geom_ribbon(data = pred_pre, aes(ymin = conf.low, ymax = conf.high, fill = task), alpha = 0.8) +
  theme_bw() +
  theme(axis.text.x = element_text(family = "Arial", colour = "black", size = 12),
        axis.text.y = element_text(family = "Arial", colour = "black", size = 12),
        axis.title.x = element_text(family = "Arial", colour = "black", size = 12),
        axis.title.y = element_text(family = "Arial", colour = "black", size = 12),
        legend.text = element_text(family = "Arial", colour = "black", size = 12),
        legend.title = element_blank(), 
        legend.position = "top") +
  scale_y_continuous(name = "Predicted Retrieval Accuracy", limits = c(0, 1), 
                     breaks = seq(0, 1, by = .2), labels = seq(0, 1, by = .2)) +
  scale_x_continuous(name = "Age (Standardized)") +
  scale_colour_manual(values = c("#8E0B99", "#617EC8"), labels = c("Easy", "Hard")) +
  scale_fill_manual(values = c("#8E0B99", "#617EC8"), labels = c("Easy", "Hard"))

# Save the figure in svg format, which we can edit in Inkscape.
ggsave("premeno_taskAge_sept2023.svg", pred_pre_fig, width = 4, height = 4)

# Conduct linear mixed-effects analysis for post-menopausal females.
lmm_post <- lmerTest::lmer(cs_rate ~ task * s2_age_std + (1|id), data = post_long)

# Generate outputs for the post-menopausal females.
anova(lmm_post)

# Generate predictions for the post-menopausal model and store.
pred_post <- ggpredict(lmm_post, terms = c( "s2_age_std", "task"))

# Amend the predictions data set for later plotting.
pred_post <- pred_post %>% 
  rename(s2_age_std = x, cs_rate = predicted, task = group)

# Plot a figure based on the predicted values for post-menopausal females.
set.seed(123) 
pred_post_fig <- pred_post %>%
  ggplot(aes(x = s2_age_std, y = cs_rate, colour = task)) +
  geom_line(aes(x = s2_age_std, y = cs_rate, colour = task)) +
  geom_ribbon(data = pred_post, aes(ymin = conf.low, ymax = conf.high, fill = task), alpha = 0.8) +
  theme_bw() +
  theme(axis.text.x = element_text(family = "Arial", colour = "black", size = 12),
        axis.text.y = element_text(family = "Arial", colour = "black", size = 12),
        axis.title.x = element_text(family = "Arial", colour = "black", size = 12),
        axis.title.y = element_text(family = "Arial", colour = "black", size = 12),
        legend.text = element_text(family = "Arial", colour = "black", size = 12),
        legend.title = element_blank(), 
        legend.position = "top") +
  scale_y_continuous(name = "Predicted Retrieval Accuracy", limits = c(0, 1), 
                     breaks = seq(0, 1, by = .2), labels = seq(0, 1, by = .2)) +
  scale_x_continuous(name = "Age (Standardized)") +
  scale_colour_manual(values = c("#8E0B99", "#617EC8"), labels = c("Easy", "Hard")) +
  scale_fill_manual(values = c("#8E0B99", "#617EC8"), labels = c("Easy", "Hard"))

# Save the figure in svg format, which we can edit in Inkscape.
ggsave("postmeno_taskAge_sept2023.svg", pred_post_fig, width = 4, height = 4)


# RUN PLS REGRESSION FOR RT DATA ------------------------------------------

# Create a new tibble, x_rt, that contains the independent variables (a.k.a., the 
# predictor variables) for the RT analysis. In this case, the independent 
# variables are menopause status at session 2 (`s2_meno_group`) and age (years) 
# at session 2 (`s2_age_std`). 
x_rt <- bhamm %>% select(s2_meno_group, s2_age_std)

# Create a new tibble, y_rt, that contains the dependent variables (a.k.a., the outcome 
# variables) for the RT analysis. In this case, the dependent variables are 
# correct source RT - easy (`rt_cs_easy`), correct source RT - hard (`rt_cs_hard`), 
# recognition RT - easy (`rt_recog_easy`), and recognition RT - hard (`rt_recog_hard`). 
y_rt <- bhamm %>% select(rt_cs_easy, rt_cs_hard, rt_recog_easy, rt_recog_hard)

# Identify and remove two participants (issue raised separately) who provided no 
# correct recognition judgments for the easy task. These participants do not, by 
# definition, have RT data for this condition, and `plsreg2` does not allow missing data.
which(is.na(y_rt$rt_recog_easy)) # 10, 44
x_rt <- x_rt[-c(10, 44), ]
y_rt <- y_rt[-c(10, 44), ]

# Run PLS regression analysis on the RT data using `plsreg2`.
set.seed(123) 
plsreg2_rt <- plsreg2(predictors = x_rt, responses = y_rt, comps = NULL, crosval = TRUE) 

# Examine the Q2 values for the LVs (a.k.a., components). Values >= 0.0975 are 
# usually considered "significant".
plsreg2_rt$Q2

# Examine the percentage of variance in X (age, menopause) and Y (performance measures) 
# explained by the LVs.
plsreg2_rt$expvar
