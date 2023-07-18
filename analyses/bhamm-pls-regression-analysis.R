# Name: bhamm-pls-regression-analysis.R
# Author: Rikki Lissaman
# Last Updated: July 18, 2023
#
# Description: This R script contains the code used to analyze data for "The 
# interactive influence of chronological age and menopause status on the functional
# neural correlates of spatial context memory in middle-aged females" by Crestol
# et al. (2023).
# 
# Inputs: 
# (1) 1 SPSS file (.sav) containing relevant data from the Brain Health at Midlife 
# and Menopause (BHAMM) study (BHAM_N72_MA_31Pre41Post_MRIsample_CorrectedFA_RT.sav)
#
# Outputs:
# N/A
#
# Additional Information:
# - Run using R version 4.1.3 (2022-03-10) on MacBook Pro.
# - The plot generated below was amended outside of R (using Inkscape) to improve
#   readability. 


# LOAD PACKAGES -----------------------------------------------------------

# Load packages for data wrangling/analysis
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

# Create a new tibble, X, that contains the independent variables (a.k.a., the 
# predictor variables). In this case, the independent variables are menopause 
# status at session 2 (`s2_meno_group`) and age (years) at session 2 (`s2_age`). 
X <- bhamm %>% select(s2_meno_group, s2_age)

# Standardize (i.e., z-score) age. Ensure it is stored as a numeric variable.
X <- X %>% 
  mutate(s2_age = scale(s2_age, center = TRUE, scale = TRUE)) %>% 
  mutate(s2_age = as.numeric(s2_age))

# Create a new tibble, Y, that contains the dependent variables (a.k.a., the outcome 
# variables). In this case, the dependent variables are correct source rate - easy 
# (`cs_rate_easy`), correct source rate - hard (`cs_rate_hard`), recognition rate 
# - easy (`recog_rate_easy`), and recognition rate - hard (`recog_rate_hard`). 
Y <- bhamm %>% select(cs_rate_easy, cs_rate_hard, recog_rate_easy, recog_rate_hard)

# Standardize (i.e., z-score) all of the dependent variables. Ensure all are stored 
# as numeric variables.
Y <- Y %>% 
  mutate_if(is.numeric, ~scale(., center = TRUE, scale = TRUE)) %>% 
  mutate_if(is.numeric, ~as.numeric(.))


# RUN PLS REGRESSION ------------------------------------------------------

# Run a PLS regression analysis using `plsreg2`. This performs a PLS2 analysis; 
# essentially, a PLS regression in the multivariate (or multi-response) case.
plsreg2_model <- plsreg2(predictors = X, responses = Y, comps = 2, crosval = TRUE) # comps = PLS components (default = 2)

# First and foremost, examine the percentage of variance in X (age, menopause) 
# and Y (performance measures) explained by the two components.
round(plsreg2_model$expvar, 4)

# Generate a loading plot for the model (similar to PCA).
plot(plsreg2_model, what = "variables", show.names = TRUE, xlab = "First Component", 
     ylab = "Second Component", main = NULL, cex = 1, pos = 4)

# Print the loadings of the predictor variables
plsreg2_model$x.loads

# Print the loadings of the outcome variables.
plsreg2_model$y.loads