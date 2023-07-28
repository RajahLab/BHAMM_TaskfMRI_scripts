# Name: bhamm-pls-regression-analysis.R
# Author: Rikki Lissaman
# Last Updated: July 28, 2023
#
# Description: This R script contains the code used to analyze data for "The 
# interactive influence of chronological age and menopause status on the functional
# neural correlates of spatial context memory in middle-aged females" by Crestol
# et al. (2023).
# 
# Inputs: 
# (1) 1 SPSS file (.sav) containing relevant data from the Brain Health at Midlife 
# and Menopause (BHAMM) study (BHAM_N72_MA_31Pre41Post_MRIsample_CorrectedFA_RT.sav).
#
# Outputs:
# N/A
#
# Additional Information:
# - Run using R version 4.1.3 (2022-03-10) on 2021 MacBook Pro (M1 chip).
# - The plot generated below was amended outside of R (using Inkscape) to improve
#   readability for publication. 


# LOAD PACKAGES -----------------------------------------------------------

# Load packages for data wrangling/analysis.
library(haven) # version 2.5.0
library(janitor) # version 2.1.0
library(tidyr) # version 1.2.0
library(dplyr) # version 1.0.8
library(plsdepot) # version 0.2.0


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


# RUN PLS REGRESSION FOR ACCURACY DATA ------------------------------------

# Create a new tibble, x_acc, that contains the independent variables (a.k.a., the 
# predictor variables) for the accuracy analysis. In this case, the independent 
# variables are menopause status at session 2 (`s2_meno_group`) and age (years) 
# at session 2 (`s2_age`). 
x_acc <- bhamm %>% select(s2_meno_group, s2_age)

# Standardize (i.e., z-score) age. Ensure it is stored as a numeric variable.
x_acc <- x_acc %>% 
  mutate(s2_age = scale(s2_age, center = TRUE, scale = TRUE)) %>% 
  mutate(s2_age = as.numeric(s2_age))

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

# Generate a correlation circle  plot for the model (similar to PCA).
plot(plsreg2_acc, what = "variables", show.names = TRUE, xlab = "LV1", 
     ylab = "LV2", main = NULL, cex = 1, pos = 4)


# RUN PLS REGRESSION FOR RT DATA ------------------------------------------

# Create a new tibble, x_rt, that contains the independent variables (a.k.a., the 
# predictor variables) for the RT analysis. In this case, the independent 
# variables are menopause status at session 2 (`s2_meno_group`) and age (years) 
# at session 2 (`s2_age`). 
x_rt <- bhamm %>% select(s2_meno_group, s2_age)

# Standardize (i.e., z-score) age. Ensure it is stored as a numeric variable.
x_rt <- x_rt %>% 
  mutate(s2_age = scale(s2_age, center = TRUE, scale = TRUE)) %>% 
  mutate(s2_age = as.numeric(s2_age))

# Create a new tibble, y_rt, that contains the dependent variables (a.k.a., the outcome 
# variables) for the RT analysis. In this case, the dependent variables are 
# correct source RT - easy (`rt_cs_easy`), correct source RT - hard (`rt_cs_hard`), 
# recognition RT - easy (`rt_recog_easy`), and recognition RT - hard (`rt_recog_hard`). 
y_rt <- bhamm %>% select(rt_cs_easy, rt_cs_hard, rt_recog_easy, rt_recog_hard)

# Standardize (i.e., z-score) all of the dependent variables. Ensure all are stored 
# as numeric variables.
y_rt <- y_rt %>% 
  mutate_if(is.numeric, ~scale(., center = TRUE, scale = TRUE)) %>% 
  mutate_if(is.numeric, ~as.numeric(.))

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

# Generate a correlation circle  plot for the model (similar to PCA).
plot(plsreg2_rt, what = "variables", show.names = TRUE, xlab = "LV1", 
     ylab = "LV2", main = NULL, cex = 1, pos = 4)
