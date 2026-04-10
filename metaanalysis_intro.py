# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 13:19:05 2026

@author: bparadis
"""

#example code for performing a meta-analysis on dummy data

#looking at hazard ratios --> the measure comparing the rate of events between two groups
#over a time period
#HR = 1 --> no difference in survival/risk between groups
#HR > 1 --> the treatment group has a higher risk than the control
#HR < 1 --> the treatment group has a lower risk than the control
#ex./ HR = 0.5 --> treatment group has half the risk of experiencing the event (50% fewer deaths)

#DerSimonian and Laird inverse variance method to investigate heterogeneity in meta-analysis
#this method uses HRs and variability metrics (variance, confidence intervals) to create a meta-analysis
#model based on inverse variances. If a study has less variance, it is attributed a higher weight


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import PythonMeta as PMA

# Load classes
data = PMA.Data()
model = PMA.Meta()
figure = PMA.Fig()

# Define settings for fixed model w/ continuous data
settings = {
    "datatype": "CONT", # 'CATE' for binary data or 'CONT' for continuous data
    "models": "Fixed", # 'Fixed' or 'Random'
    "algorithm": "IV", # based on datatype and effect size
    "effect": "MD" # RR/OR/RD for binary data; SMD/MD for continuous data
}

#NOTE: with a fixed model type, we assume all studies estimate the same underlying 
#treatment effects and that observed differences are mostly due to sampling error 
#and natural variability. Even if differences in hospitals, populations, methods, etc the model
#assumes those differences do not produce real differences in the true treatment effect, just noise

#Random effects model would assume that each study has its own true effect, which all vary around
#an overall mean effect. If it is suspected that the differences in hospitals, populations, 
#methods, etc. might impact effect size, then random effects model is the move

#fixed effects are only appropriate when studies are extremely comparable

# Set data type
data.datatype = settings["datatype"]

# Dummy data - studies comparing two groups
# study name/HR1/variance/n subjects/HR2/variance/n subjects
samp_cont = [
    "Study1 (2005), 22.9, 6.0, 15, 27.4, 6.5, 24",
    "Study2 (2015), 7.8, 5.2, 51, 11.9, 5.3, 53",
    "Study3 (2009), 20.38, 5.26, 35, 24.32, 5.23, 35",
    "Study4 (2012), 6.67, 8.28, 43, 12.67, 9.87, 43",
    "Study5 (2001), 18.49, 7.16, 50, 20.72, 8.67, 60",
    "Study6 (2010), 11.8, 5.7, 40, 13.0, 5.2, 40",
    "Study7 (2016), 6.8, 4.7, 40, 8.0, 4.2, 45",
    "Study8 (2011), 12.9, 2.7, 40, 9.2, 5.2, 40"
]

# Load data
studies = data.getdata(samp_cont)

# Set the subgroup, data type, models, algorithm, and effect for metaanalysis
model.subgroup = data.subgroup
model.datatype = data.datatype
model.models = settings["models"]
model.algorithm = settings["algorithm"]
model.effect = settings["effect"]

# Perform the meta analysis
results = model.meta(studies)

# call the figure PMA class defined in first block to make meta-analysis figs
#forest plot
figure.forest(results).show()
# Show funnel plot
figure.funnel(results).show()

#interpretation:
    # forest plot diamond shows the experimental treatment is favored
    # entire diamond is to the left of the reference line, so statistically significant result
    
    #funnel plot shows symmetry of the studies' effect relative to the reference line
    #high symmetry expected when both positive and negative effects are published
    #looking for asymmetry which would indicate publication bias
    #publication bias occurs if studies with statistically significant effect sizes are
    #highly likely selected and published than studies with no significant effect sizes
    #pub bias can lead to overestimate of true effect size
    
    #Egger's test puts a p-value on determining pub bias --> if signif <0.05, pub bias present

    #smaller studies will have widely scattered effect size estimates and higher variance
    #larger studies will have higher precision, clustered near the top with lower variance
    
###################
#subgroup analysis#
###################
#are there any subgroups in the data -- subjects who share common characteristics that
#differentiates them biologically from the rest of the population
#ex./ mutation, blood type, gene expression

# Define settings for subgroup analysis w/ random effect & binary data
settings = {
    "datatype": "CATE", # 'CATE' for binary data or 'CONT' for continuous data
    "models": "Random", # 'Fixed' or 'Random'
    "algorithm": "IV", # based on datatype and effect size
    "effect": "RR" # RR (risk ratio)/OR/RD for binary data; SMD/MD for continuous data
}
# Set data type
data.datatype = settings["datatype"]

# Provided data with subgroup    
#study name/events_treatment/events_control/total_control
samp_cate = [
    "Study1 (2005), 15, 30, 8, 30",
    "Study2 (2015), 20, 50, 10, 50",
    "Study3 (2009), 25, 70, 15, 70",
    "Study4 (2012), 10, 50, 20, 50",
    "<subgroup>name=Mutation_A",
    "Study5 (2001), 30, 110, 25, 110",
    "Study6 (2010), 15, 80, 80, 80",
    "Study7 (2016), 15, 85, 60, 90",
    "Study8 (2011), 20, 90, 30, 90",
    "<subgroup>name=Mutation_B"
]

# Load data
studies = data.getdata(samp_cate)

# Set the subgroup, data type, models, algorithm, and effect for model of the metaanalysis
model.subgroup = data.subgroup
model.datatype = data.datatype
model.models = settings["models"]
model.algorithm = settings["algorithm"]
model.effect = settings["effect"]

# Perform the analysis
results = model.meta(studies)
# Show forest plot 
figure.forest(results).show()
# Show funnel plot
figure.funnel(results).show()

#interpretation:
    # forest plot has three diamonds - one for each subgroup & one for overall effect
    #overall effect is slightly below 1, so treatment effect is present
    #but treatment effect is stronger with subgroup B
    #subgroup A diamond is above 1, so the treatment only really worked for group B
    
    
#########################
#Meta-regression example#
#########################

#how do differnt variables affect the overall meta-analysis? 
#ex./ how does age affect a therapy or is a specific biomarker associated with results we see?

#can consider a specific variable (moderator) to see whether it affects the meta-analysis results
# with a meta-regression 
# ex./ if patients differe in age from study to study, then age can influence how a treatment
# works across different studies
 
# Create a dummy dataset (with normal distribution), random seed for reproducibility
np.random.seed(0)
n_studies = 10
effect_sizes = np.random.normal(loc=0.2, scale=0.1, size=n_studies)
variances = np.random.uniform(low=0.01, high=0.05, size=n_studies)
weights = 1 / variances
covariates = np.random.normal(loc=0, scale=1, size=n_studies)

# Create a DataFrame
df = pd.DataFrame({
    'effect_sizes': effect_sizes,
    'variances': variances,
    'weights': weights,
    'covariates': covariates
})
#Add an intercept. This is needed in modeling the regression
df['intercept'] = 1

# Perform weighted least squares regression
model = sm.WLS(df['effect_sizes'],
               df[['intercept', 'covariates']],
               weights=df['weights'])
results = model.fit()

# Print the results
print("Coefficients:")
print(results.params)
print("\nConfidence Intervals:")
print(results.conf_int())
print("\nP-values:")
print(results.pvalues)
print("\nStandard Errors:")
print(results.bse)

#Visualize w/ a bubble plot
# Create a bubble plot without confidence intervals
plt.figure(figsize=(10, 6))
plt.scatter(df['covariates'],
            df['effect_sizes'], s=100 * df['weights'],
            alpha=0.5, edgecolors='black')
plt.plot(df['covariates'], results.fittedvalues, 'r-')
plt.xlabel('Covariate')
plt.ylabel('Effect Size')
plt.title('Bubble Plot with Trendline')
plt.grid(True)
plt.show()

#interpretation:
    # explains association between a moderator and the effect size
    # trendline can be used to explain how the main effect varies across different values of the 
    # moderator/covariate (ie, age)
    # each bubble represents a single study; the size of the bubble is inverse to study's variability
    # ex./ larger bubble = study w/ high confidence, low variance, low uncertainty